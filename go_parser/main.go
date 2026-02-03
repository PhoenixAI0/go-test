package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
)

// Sample represents a single training sample
type Sample struct {
	Prompt    string `json:"prompt"`
	Response  string `json:"response"`
	Player    string `json:"player"`
	BoardSize int    `json:"board_size"`
}

// SGFGame represents a parsed SGF game
type SGFGame struct {
	BoardSize int
	Moves     []Move
}

// Move represents a single move in the game
type Move struct {
	Player string
	Row    int
	Col    int
	IsPass bool
}

var (
	sizeRegex  = regexp.MustCompile(`SZ\[(\d+)\]`)
	blackRegex = regexp.MustCompile(`;B\[([a-s]{2}|)\]`)
	whiteRegex = regexp.MustCompile(`;W\[([a-s]{2}|)\]`)
)

// parseSGF parses an SGF file and returns a game
func parseSGF(filename string) (*SGFGame, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	content := string(data)

	// Extract board size
	boardSize := 19
	if match := sizeRegex.FindStringSubmatch(content); len(match) > 1 {
		fmt.Sscanf(match[1], "%d", &boardSize)
	}

	game := &SGFGame{
		BoardSize: boardSize,
		Moves:     make([]Move, 0),
	}

	// Parse black moves
	blackMatches := blackRegex.FindAllStringSubmatch(content, -1)
	whiteMatches := whiteRegex.FindAllStringSubmatch(content, -1)

	// Interleave moves (simplified - assumes alternating play)
	maxLen := len(blackMatches)
	if len(whiteMatches) > maxLen {
		maxLen = len(whiteMatches)
	}

	for i := 0; i < maxLen; i++ {
		// Black move
		if i < len(blackMatches) {
			coord := blackMatches[i][1]
			move := Move{Player: "B"}
			if coord == "" {
				move.IsPass = true
			} else {
				move.Row = int(coord[1] - 'a')
				move.Col = int(coord[0] - 'a')
			}
			game.Moves = append(game.Moves, move)
		}

		// White move
		if i < len(whiteMatches) {
			coord := whiteMatches[i][1]
			move := Move{Player: "W"}
			if coord == "" {
				move.IsPass = true
			} else {
				move.Row = int(coord[1] - 'a')
				move.Col = int(coord[0] - 'a')
			}
			game.Moves = append(game.Moves, move)
		}
	}

	return game, nil
}

// coordsToGTP converts board coordinates to GTP format
func coordsToGTP(row, col, boardSize int) string {
	if row < 0 || row >= boardSize || col < 0 || col >= boardSize {
		return "PASS"
	}

	// GTP uses letters A-T, skipping I
	letter := rune('A' + col)
	if letter >= 'I' {
		letter++
	}

	// GTP rows are numbered from bottom (1) to top (boardSize)
	number := boardSize - row

	return fmt.Sprintf("%c%d", letter, number)
}

// generateBoardPrompt creates a simple board state prompt
func generateBoardPrompt(boardSize int, moveNum int) string {
	// Simplified: in production, this should track actual board state
	// For now, just indicate board size and move number
	return fmt.Sprintf("Board size: %dx%d, Move: %d", boardSize, boardSize, moveNum)
}

// processSGFFile processes a single SGF file and returns samples
func processSGFFile(filename string, maxSamples int) ([]Sample, error) {
	game, err := parseSGF(filename)
	if err != nil {
		return nil, err
	}

	samples := make([]Sample, 0)
	for i, move := range game.Moves {
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}

		response := "PASS"
		if !move.IsPass {
			response = coordsToGTP(move.Row, move.Col, game.BoardSize)
		}

		sample := Sample{
			Prompt:    generateBoardPrompt(game.BoardSize, i),
			Response:  response,
			Player:    move.Player,
			BoardSize: game.BoardSize,
		}

		samples = append(samples, sample)
	}

	return samples, nil
}

// worker processes SGF files from a channel
func worker(id int, jobs <-chan string, results chan<- []Sample, maxSamplesPerFile int, wg *sync.WaitGroup) {
	defer wg.Done()

	for filename := range jobs {
		samples, err := processSGFFile(filename, maxSamplesPerFile)
		if err != nil {
			log.Printf("Worker %d: Error processing %s: %v", id, filename, err)
			continue
		}
		if len(samples) > 0 {
			results <- samples
		}
	}
}

func main() {
	sgfDir := flag.String("sgf-dir", "", "Directory containing SGF files")
	outputFile := flag.String("output", "", "Output JSON file")
	maxPositions := flag.Int("max-positions", 200000, "Maximum positions to extract")
	numWorkers := flag.Int("workers", runtime.NumCPU(), "Number of parallel workers")
	flag.Parse()

	if *sgfDir == "" || *outputFile == "" {
		log.Fatal("Usage: sgf_parser -sgf-dir <dir> -output <file>")
	}

	// Find all SGF files
	var sgfFiles []string
	err := filepath.Walk(*sgfDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".sgf") {
			sgfFiles = append(sgfFiles, path)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Found %d SGF files", len(sgfFiles))
	log.Printf("Using %d workers", *numWorkers)

	// Create output file
	outFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024) // 1MB buffer
	defer writer.Flush()

	// Create channels
	jobs := make(chan string, 100)
	results := make(chan []Sample, 100)

	// Start workers
	var wg sync.WaitGroup
	maxSamplesPerFile := (*maxPositions / len(sgfFiles)) + 1

	for w := 0; w < *numWorkers; w++ {
		wg.Add(1)
		go worker(w, jobs, results, maxSamplesPerFile, &wg)
	}

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Send jobs
	go func() {
		for _, file := range sgfFiles {
			jobs <- file
		}
		close(jobs)
	}()

	// Collect results
	totalSamples := 0
	for samples := range results {
		for _, sample := range samples {
			if totalSamples >= *maxPositions {
				break
			}

			jsonData, err := json.Marshal(sample)
			if err != nil {
				log.Printf("Error marshaling sample: %v", err)
				continue
			}

			writer.Write(jsonData)
			writer.WriteByte('\n')
			totalSamples++

			if totalSamples%10000 == 0 {
				log.Printf("Processed %d samples...", totalSamples)
			}
		}
		if totalSamples >= *maxPositions {
			break
		}
	}

	log.Printf("Completed: wrote %d samples to %s", totalSamples, *outputFile)
}
