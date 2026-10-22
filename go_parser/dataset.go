package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// Sample represents a single training sample.
type Sample struct {
	Prompt           string      `json:"prompt"`
	Response         string      `json:"response"`
	OpponentResponse *string     `json:"opponent_response"`
	BoardSize        int         `json:"board_size"`
	GameID           string      `json:"game_id"`
	SourceFile       string      `json:"source_file"`
	MoveNumber       int         `json:"move_number"`
	ToPlay           string      `json:"to_play"`
	InitialPlayer    string      `json:"initial_player"`
	Komi             *float64    `json:"komi"`
	Rules            string      `json:"rules"`
	Result           string      `json:"result"`
	Handicap         int         `json:"handicap"`
	InitialStones    [][2]string `json:"initial_stones"`
	Moves            [][2]string `json:"moves"`
}

func buildDataset(sgfDir string, outputPath string, maxPositions int, workers int) (count int, err error) {
	return buildDatasetWithOptions(sgfDir, outputPath, maxPositions, workers, 42, 0)
}

func buildDatasetWithOptions(sgfDir string, outputPath string, maxPositions int, workers int, seed int64, positionsPerGame int) (count int, err error) {
	if maxPositions < 0 || positionsPerGame < 0 {
		return 0, fmt.Errorf("position limits must be nonnegative")
	}
	files, err := listSgfFiles(sgfDir)
	if err != nil {
		return 0, err
	}
	if len(files) == 0 {
		log.Printf("No SGF files found in %s", sgfDir)
	}
	// Shuffle whole games, rather than always taking the lexicographic prefix of
	// a source archive. A local RNG keeps sampling reproducible across workers.
	rand.New(rand.NewSource(seed)).Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })

	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	runtime.GOMAXPROCS(workers)

	// Always create the output so callers can read an empty dataset.
	outFile, err := os.Create(outputPath)
	if err != nil {
		return 0, err
	}
	defer func() {
		// A failed close can mean lost data; never report a truncated file as success.
		if closeErr := outFile.Close(); err == nil {
			err = closeErr
		}
	}()

	writer := bufio.NewWriterSize(outFile, 1024*1024)
	count, err = writeParallelOptions(files, writer, maxPositions, max(1, workers), seed, positionsPerGame, sgfDir)
	if err != nil {
		return count, err
	}
	return count, writer.Flush()
}

func listSgfFiles(sgfDir string) ([]string, error) {
	files := make([]string, 0)
	err := filepath.WalkDir(sgfDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(path), ".sgf") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	return files, nil
}

func writeSequential(files []string, writer *bufio.Writer, maxPositions int) (int, error) {
	return writeParallelOptions(files, writer, maxPositions, 1, 42, 0, "")
}

func writeParallel(files []string, writer *bufio.Writer, maxPositions int, workers int) (int, error) {
	return writeParallelOptions(files, writer, maxPositions, workers, 42, 0, "")
}

func writeParallelOptions(files []string, writer *bufio.Writer, maxPositions int, workers int, seed int64, positionsPerGame int, sgfDir string) (int, error) {
	batchSize := max(1, workers*4)
	collected := 0
	seen := make(map[string]bool)

	for batchStart := 0; batchStart < len(files); batchStart += batchSize {
		if collected >= maxPositions {
			break
		}
		batchEnd := min(batchStart+batchSize, len(files))
		batchFiles := files[batchStart:batchEnd]
		remaining := maxPositions - collected

		results := make([][]Sample, len(batchFiles))
		var wg sync.WaitGroup
		sem := make(chan struct{}, workers)

		for i, path := range batchFiles {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, filePath string) {
				defer wg.Done()
				defer func() { <-sem }()
				// Extract whole games before subsampling, so late middlegame and
				// endgame positions survive small per-game budgets.
				samples, err := processSGFFile(filePath, int(^uint(0)>>1))
				if err != nil {
					log.Printf("Skipping SGF file %s: %v", filePath, err)
					return
				}
				if positionsPerGame > 0 && len(samples) > positionsPerGame {
					digest := sha256.Sum256([]byte(fmt.Sprintf("%d:%s", seed, samples[0].GameID)))
					var gameSeed int64
					for _, b := range digest[:8] {
						gameSeed = gameSeed<<8 | int64(b)
					}
					indices := rand.New(rand.NewSource(gameSeed)).Perm(len(samples))[:positionsPerGame]
					sort.Ints(indices)
					selected := make([]Sample, 0, positionsPerGame)
					for _, index := range indices {
						selected = append(selected, samples[index])
					}
					samples = selected
				}
				if len(samples) > remaining {
					samples = samples[:remaining]
				}
				for j := range samples {
					rel, relErr := filepath.Rel(sgfDir, filePath)
					if relErr == nil {
						samples[j].SourceFile = filepath.ToSlash(rel)
					} else {
						samples[j].SourceFile = filepath.Base(filePath)
					}
				}
				results[idx] = samples
			}(i, path)
		}
		wg.Wait()

		for _, samples := range results {
			if collected >= maxPositions {
				break
			}
			if len(samples) == 0 {
				continue
			}
			// Game IDs ignore annotations (including result), so archive copies
			// and rotated copies stay grouped. Distinct rules/komi retain labels.
			identity, _ := json.Marshal([]any{samples[0].GameID, samples[0].Rules, samples[0].Komi})
			if seen[string(identity)] {
				continue
			}
			seen[string(identity)] = true
			for _, sample := range samples {
				if collected >= maxPositions {
					break
				}
				if err := writeSample(writer, sample); err != nil {
					return collected, err
				}
				collected++
				if collected%10000 == 0 {
					log.Printf("Processed %d samples...", collected)
				}
			}
		}
	}
	return collected, nil
}

func processSGFFile(path string, maxPositions int) ([]Sample, error) {
	game, err := parseSGFFile(path)
	if err != nil {
		return nil, err
	}
	return samplePositions(game, maxPositions)
}

func samplePositions(game *SgfGame, maxPositions int) ([]Sample, error) {
	board := NewBoardState(game.BoardSize)
	capacity := maxPositions
	if capacity > len(game.Moves) {
		capacity = len(game.Moves)
	}
	if capacity < 0 {
		capacity = 0
	}
	samples := make([]Sample, 0, capacity)
	if maxPositions <= 0 {
		return samples, nil
	}
	initial := make([][2]string, 0, len(game.SetupStones))
	history := make([][2]string, 0, len(game.Moves))
	gameID := gameFingerprint(game)

	for _, stone := range game.SetupStones {
		gtp, err := coordsToGTP(stone.Coords, game.BoardSize)
		if err != nil {
			return nil, err
		}
		if stone.Coords == nil {
			return nil, fmt.Errorf("setup stone cannot be PASS")
		}
		idx := stone.Coords.Row*game.BoardSize + stone.Coords.Col
		if board.board[idx] != stoneEmpty {
			return nil, fmt.Errorf("duplicate setup stone: %s", gtp)
		}
		color := byte(stoneBlack)
		if stone.Player == "W" {
			color = stoneWhite
		} else if stone.Player != "B" {
			return nil, fmt.Errorf("invalid setup player: %s", stone.Player)
		}
		board.board[idx] = color
		initial = append(initial, [2]string{stone.Player, gtp})
	}

	for index, move := range game.Moves {
		expectedPlayer := game.InitialPlayer
		if index > 0 {
			expectedPlayer = "B"
			if game.Moves[index-1].Player == "B" {
				expectedPlayer = "W"
			}
		}
		if expectedPlayer != "" && move.Player != expectedPlayer {
			return nil, fmt.Errorf("nonalternating move at turn %d", index)
		}
		gtp, err := coordsToGTP(move.Coords, game.BoardSize)
		if err != nil {
			return nil, err
		}
		prompt, err := board.Prompt(move.Player)
		if err != nil {
			return nil, err
		}
		// Validate/apply before adding the label. Imported games can contain
		// suicide or ko moves under rules that differ from our training rules.
		if err := board.ApplyMove(move.Player, gtp, true); err != nil {
			log.Printf("Stopping game before illegal training move %s: %v", gtp, err)
			break
		}
		if len(samples) > 0 {
			// Predict the next recorded opponent move as an optional auxiliary
			// target, following Darkforest/KataGo. Never supervise an illegal
			// future label: it is attached only after that move validates.
			nextMove := gtp
			samples[len(samples)-1].OpponentResponse = &nextMove
		}
		samples = append(samples, Sample{
			Prompt:        prompt,
			Response:      gtp,
			BoardSize:     game.BoardSize,
			GameID:        gameID,
			MoveNumber:    index,
			ToPlay:        move.Player,
			InitialPlayer: game.InitialPlayer,
			Komi:          game.Komi,
			Rules:         game.Rules,
			Result:        game.Result,
			Handicap:      game.Handicap,
			InitialStones: initial,
			Moves:         append(make([][2]string, 0, len(history)), history...),
		})
		history = append(history, [2]string{move.Player, gtp})
		if len(samples) >= maxPositions {
			break
		}
	}

	return samples, nil
}

// Canonicalize all eight symmetries before hashing. Comments, filenames, and
// winner annotations never create different train/test groups for one game.
func gameFingerprint(game *SgfGame) string {
	canonical := ""
	for symmetry := 0; symmetry < 8; symmetry++ {
		encode := func(move Move) string {
			if move.Coords == nil {
				return move.Player + ":pass"
			}
			r, c := move.Coords.Row, move.Coords.Col
			if symmetry >= 4 {
				c = game.BoardSize - 1 - c
			}
			for k := 0; k < symmetry%4; k++ {
				r, c = c, game.BoardSize-1-r
			}
			return fmt.Sprintf("%s:%d,%d", move.Player, r, c)
		}
		setup := make([]string, 0, len(game.SetupStones))
		for _, stone := range game.SetupStones {
			setup = append(setup, encode(stone))
		}
		sort.Strings(setup)
		moves := make([]string, 0, len(game.Moves))
		for _, move := range game.Moves {
			moves = append(moves, encode(move))
		}
		representation := fmt.Sprintf("%d|%s|%s", game.BoardSize, strings.Join(setup, ";"), strings.Join(moves, ";"))
		if symmetry == 0 || representation < canonical {
			canonical = representation
		}
	}
	return fmt.Sprintf("%x", sha256.Sum256([]byte(canonical)))
}

// Go string quoting is not JSON (for example \x1b); use encoding/json.
func writeSample(writer *bufio.Writer, sample Sample) error {
	data, err := json.Marshal(sample)
	if err != nil {
		return err
	}
	_, err = writer.Write(append(data, '\n'))
	return err
}
