package main

import (
	"bufio"
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestSamplePositionsPromptIncludesSetup(t *testing.T) {
	game := &SgfGame{
		BoardSize: 5,
		SetupStones: []Move{
			{Player: "B", Coords: &Coord{Row: 2, Col: 2}},
		},
		Moves: []Move{
			{Player: "B", Coords: &Coord{Row: 4, Col: 0}},
			{Player: "W", Coords: nil},
		},
	}
	samples, err := samplePositions(game, 2)
	if err != nil {
		t.Fatalf("samplePositions: %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(samples))
	}
	if samples[0].Response != "A1" {
		t.Fatalf("expected response A1, got %s", samples[0].Response)
	}
	if !bytes.Contains([]byte(samples[0].Prompt), []byte("..X..")) {
		t.Fatalf("expected prompt to include setup stone at C3")
	}
	if !bytes.Contains([]byte(samples[0].Prompt), []byte("to_play: B")) {
		t.Fatalf("expected prompt to include to_play header")
	}
}

func TestWriteSampleJSON(t *testing.T) {
	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)
	prompt := "board_size: 5\nto_play: B\nboard:\n.....\n.....\n.....\n.....\n.....\nmove:"
	if err := writeSample(writer, Sample{Prompt: prompt, Response: "A1", BoardSize: 5}); err != nil {
		t.Fatalf("writeSample: %v", err)
	}
	if err := writer.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	expected := "{\"prompt\": \"board_size: 5\\nto_play: B\\nboard:\\n.....\\n.....\\n.....\\n.....\\n.....\\nmove:\", \"response\": \"A1\", \"board_size\": 5}\n"
	if buf.String() != expected {
		t.Fatalf("unexpected JSONL output: %s", buf.String())
	}
}

func TestListSgfFilesFiltersAndSorts(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "nested")
	if err := os.MkdirAll(nested, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	files := []string{
		filepath.Join(dir, "b.sgf"),
		filepath.Join(dir, "a.sgf"),
		filepath.Join(nested, "c.sgf"),
	}
	for _, path := range files {
		if err := os.WriteFile(path, []byte("(;SZ[5])"), 0644); err != nil {
			t.Fatalf("write file: %v", err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignore"), 0644); err != nil {
		t.Fatalf("write txt: %v", err)
	}

	listed, err := listSgfFiles(dir)
	if err != nil {
		t.Fatalf("listSgfFiles: %v", err)
	}
	if len(listed) != 3 {
		t.Fatalf("expected 3 sgf files, got %d", len(listed))
	}
	expected := []string{
		filepath.Join(dir, "a.sgf"),
		filepath.Join(dir, "b.sgf"),
		filepath.Join(nested, "c.sgf"),
	}
	for i, path := range expected {
		if listed[i] != path {
			t.Fatalf("expected %s at %d, got %s", path, i, listed[i])
		}
	}
}

func TestSamplePositionsStopsOnIllegalMove(t *testing.T) {
	game := &SgfGame{
		BoardSize: 5,
		Moves: []Move{
			{Player: "B", Coords: &Coord{Row: 0, Col: 0}},
			{Player: "W", Coords: &Coord{Row: 0, Col: 0}},
			{Player: "B", Coords: &Coord{Row: 1, Col: 1}},
		},
	}
	samples, err := samplePositions(game, 10)
	if err != nil {
		t.Fatalf("samplePositions: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("expected only the legal move's sample, got %d", len(samples))
	}
	if samples[0].Response != "A5" {
		t.Fatalf("unexpected response: %s", samples[0].Response)
	}
}

func TestBuildDatasetSequentialWritesJSONL(t *testing.T) {
	dir := t.TempDir()
	sgfDir := filepath.Join(dir, "sgf")
	if err := os.MkdirAll(sgfDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	sgfPath := filepath.Join(sgfDir, "game.sgf")
	sgfContent := "(;SZ[5];B[aa];W[bb];B[cc])"
	if err := os.WriteFile(sgfPath, []byte(sgfContent), 0644); err != nil {
		t.Fatalf("write sgf: %v", err)
	}
	outputPath := filepath.Join(dir, "out.jsonl")

	count, err := buildDataset(sgfDir, outputPath, 2, 1)
	if err != nil {
		t.Fatalf("buildDataset: %v", err)
	}
	if count != 2 {
		t.Fatalf("expected 2 samples, got %d", count)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d", len(lines))
	}
	if !bytes.Contains(lines[0], []byte("\"board_size\": 5")) {
		t.Fatalf("expected board_size in output line: %s", string(lines[0]))
	}
}
