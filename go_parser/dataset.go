package main

import (
	"bufio"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Sample represents a single training sample.
type Sample struct {
	Prompt    string `json:"prompt"`
	Response  string `json:"response"`
	BoardSize int    `json:"board_size"`
}

func buildDataset(sgfDir string, outputPath string, maxPositions int, workers int) (int, error) {
	if maxPositions <= 0 {
		return 0, nil
	}

	files, err := listSgfFiles(sgfDir)
	if err != nil {
		return 0, err
	}
	if len(files) == 0 {
		log.Printf("No SGF files found in %s", sgfDir)
		return 0, nil
	}

	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	runtime.GOMAXPROCS(workers)

	outFile, err := os.Create(outputPath)
	if err != nil {
		return 0, err
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024)
	defer writer.Flush()

	useParallel := workers > 1 && len(files) >= workers
	if useParallel {
		return writeParallel(files, writer, maxPositions, workers)
	}
	return writeSequential(files, writer, maxPositions)
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
		if strings.HasSuffix(path, ".sgf") {
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
	collected := 0
	for _, path := range files {
		if collected >= maxPositions {
			break
		}
		remaining := maxPositions - collected
		samples, err := processSGFFile(path, remaining)
		if err != nil {
			log.Printf("Skipping SGF file %s: %v", path, err)
			continue
		}
		for _, sample := range samples {
			if collected >= maxPositions {
				break
			}
			if err := writeSample(writer, sample); err != nil {
				log.Printf("Error writing sample: %v", err)
				continue
			}
			collected++
			if collected%10000 == 0 {
				log.Printf("Processed %d samples...", collected)
			}
		}
	}
	return collected, nil
}

func writeParallel(files []string, writer *bufio.Writer, maxPositions int, workers int) (int, error) {
	batchSize := max(1, workers*4)
	collected := 0

	for batchStart := 0; batchStart < len(files); batchStart += batchSize {
		if collected >= maxPositions {
			break
		}
		batchEnd := min(batchStart+batchSize, len(files))
		batchFiles := files[batchStart:batchEnd]
		remaining := maxPositions - collected
		positionsPerFile := max(1, remaining/len(batchFiles))

		results := make([][]Sample, len(batchFiles))
		var wg sync.WaitGroup
		sem := make(chan struct{}, workers)

		for i, path := range batchFiles {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, filePath string) {
				defer wg.Done()
				defer func() { <-sem }()
				samples, err := processSGFFile(filePath, positionsPerFile)
				if err != nil {
					log.Printf("Skipping SGF file %s: %v", filePath, err)
					return
				}
				results[idx] = samples
			}(i, path)
		}
		wg.Wait()

		for _, samples := range results {
			if collected >= maxPositions {
				break
			}
			for _, sample := range samples {
				if collected >= maxPositions {
					break
				}
				if err := writeSample(writer, sample); err != nil {
					log.Printf("Error writing sample: %v", err)
					continue
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

	for _, stone := range game.SetupStones {
		gtp, err := coordsToGTP(stone.Coords, game.BoardSize)
		if err != nil {
			return nil, err
		}
		if err := board.ApplyMove(stone.Player, gtp, false); err != nil {
			log.Printf("Skipping invalid setup stone %s: %v", gtp, err)
		}
	}

	for _, move := range game.Moves {
		gtp, err := coordsToGTP(move.Coords, game.BoardSize)
		if err != nil {
			return nil, err
		}
		prompt, err := board.Prompt()
		if err != nil {
			return nil, err
		}
		samples = append(samples, Sample{
			Prompt:    prompt,
			Response: gtp,
			BoardSize: game.BoardSize,
		})

		if err := board.ApplyMove(move.Player, gtp, false); err != nil {
			log.Printf("Stopping game after illegal move %s: %v", gtp, err)
			break
		}
		if len(samples) >= maxPositions {
			break
		}
	}

	return samples, nil
}

func writeSample(writer *bufio.Writer, sample Sample) error {
	prompt := strconv.QuoteToASCII(sample.Prompt)
	response := strconv.QuoteToASCII(sample.Response)
	_, err := writer.WriteString("{\"prompt\": ")
	if err != nil {
		return err
	}
	if _, err = writer.WriteString(prompt); err != nil {
		return err
	}
	if _, err = writer.WriteString(", \"response\": "); err != nil {
		return err
	}
	if _, err = writer.WriteString(response); err != nil {
		return err
	}
	if _, err = writer.WriteString(", \"board_size\": "); err != nil {
		return err
	}
	if _, err = writer.WriteString(strconv.Itoa(sample.BoardSize)); err != nil {
		return err
	}
	_, err = writer.WriteString("}\n")
	return err
}
