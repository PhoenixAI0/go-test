package main

import (
	"flag"
	"log"
	"runtime"
)

func main() {
	sgfDir := flag.String("sgf-dir", "", "Directory containing SGF files")
	outputFile := flag.String("output", "", "Output JSONL file")
	maxPositions := flag.Int("max-positions", 200000, "Maximum positions to extract")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of parallel workers")
	flag.Parse()

	if *sgfDir == "" || *outputFile == "" {
		log.Fatal("Usage: sgf_parser -sgf-dir <dir> -output <file>")
	}

	log.Printf("Scanning %s", *sgfDir)
	log.Printf("Using %d workers", *workers)

	count, err := buildDataset(*sgfDir, *outputFile, *maxPositions, *workers)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Completed: wrote %d samples to %s", count, *outputFile)
}
