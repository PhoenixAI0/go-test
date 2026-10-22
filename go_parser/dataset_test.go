package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
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
	var decoded Sample
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil || decoded.Prompt != prompt || decoded.Response != "A1" || decoded.BoardSize != 5 {
		t.Fatalf("unexpected JSONL output: %s (%v)", buf.String(), err)
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
	if !bytes.Contains(lines[0], []byte("\"board_size\":5")) {
		t.Fatalf("expected board_size in output line: %s", string(lines[0]))
	}
}

func TestDatasetIndependentOfWorkers(t *testing.T) {
	dir := t.TempDir()
	for i, moves := range []string{";B[aa];W[bb];B[cc]", ";B[ab];W[bc];B[cd]", ";B[ac];W[bd];B[ce]", ";B[ad];W[be];B[ca]", ";B[ba];W[cb];B[dc]"} {
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%d.sgf", i)), []byte("(;SZ[5]"+moves+")"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	one, many := filepath.Join(dir, "one.jsonl"), filepath.Join(dir, "many.jsonl")
	for _, settings := range []struct {
		path    string
		workers int
	}{{one, 1}, {many, 3}} {
		count, err := buildDatasetWithOptions(dir, settings.path, 7, settings.workers, 42, 2)
		if err != nil || count != 7 {
			t.Fatalf("count=%d err=%v", count, err)
		}
	}
	first, _ := os.ReadFile(one)
	second, _ := os.ReadFile(many)
	if !bytes.Equal(first, second) {
		t.Fatalf("worker count changes the dataset\n%s\n%s", first, second)
	}
}

func TestGameFingerprintIgnoresAnnotationsAndSymmetry(t *testing.T) {
	one, err := parseSGFFile(writeTempSGF(t, "(;SZ[5]KM[7.5]RU[Chinese]C[first];B[ab];W[cd];B[ee])"))
	if err != nil {
		t.Fatal(err)
	}
	two, err := parseSGFFile(writeTempSGF(t, "(;SZ[5]KM[6.5]RE[W+R]C[copy];B[ba];W[dc];B[ee])"))
	if err != nil {
		t.Fatal(err)
	}
	if gameFingerprint(one) != gameFingerprint(two) {
		t.Fatal("transposed copies must share a split group")
	}
}

func TestMetadataAndSetupArePreserved(t *testing.T) {
	game, err := parseSGFFile(writeTempSGF(t, "(;SZ[5]KM[0.5]RU[Chinese]RE[W+2.5]HA[2]PL[W]AB[aa][ee];W[cc];B[dd])"))
	if err != nil {
		t.Fatal(err)
	}
	samples, err := samplePositions(game, 10)
	if err != nil || len(samples) != 2 {
		t.Fatalf("%v samples=%d", err, len(samples))
	}
	second := samples[1]
	if second.Komi == nil || *second.Komi != 0.5 || second.Rules != "Chinese" || second.Result != "W+2.5" || second.Handicap != 2 || second.InitialPlayer != "W" || second.ToPlay != "B" || second.MoveNumber != 1 {
		t.Fatalf("lost metadata: %+v", second)
	}
	if !reflect.DeepEqual(second.Moves, [][2]string{{"W", "C3"}}) {
		t.Fatalf("wrong prefix: %v", second.Moves)
	}
	if len(second.InitialStones) != 2 || len(samples[0].Moves) != 0 {
		t.Fatal("setup mixed into move prefix")
	}
	if samples[0].OpponentResponse == nil || *samples[0].OpponentResponse != "D2" || second.OpponentResponse != nil {
		t.Fatal("opponent auxiliary target is not the next validated move")
	}
	if samples, err := samplePositions(game, 0); err != nil || len(samples) != 0 {
		t.Fatal("zero budget emitted a sample")
	}
}

func TestDuplicateGamesRemoved(t *testing.T) {
	dir := t.TempDir()
	for i, comment := range []string{"original", "copy"} {
		os.WriteFile(filepath.Join(dir, fmt.Sprintf("%d.sgf", i)), []byte("(;SZ[5]C["+comment+"];B[ab];W[cc])"), 0644)
	}
	count, err := buildDataset(dir, filepath.Join(dir, "out.jsonl"), 100, 2)
	if err != nil || count != 2 {
		t.Fatalf("expected two unique positions: %d, %v", count, err)
	}
}

func TestBuildDatasetCreatesEmptyOutputWithoutGames(t *testing.T) {
	dir := t.TempDir()
	outputPath := filepath.Join(dir, "out.jsonl")
	count, err := buildDataset(dir, outputPath, 10, 1)
	if err != nil || count != 0 {
		t.Fatalf("buildDataset: count=%d err=%v", count, err)
	}
	if data, err := os.ReadFile(outputPath); err != nil || len(data) != 0 {
		t.Fatalf("expected an empty output file, got %q (err=%v)", data, err)
	}
}

func TestWriteSampleEscapesControlCharactersAsJSON(t *testing.T) {
	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)
	if err := writeSample(writer, Sample{Prompt: "a\x1bb", Response: "A1", BoardSize: 5}); err != nil {
		t.Fatalf("writeSample: %v", err)
	}
	if err := writer.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	var decoded Sample
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("output is not valid JSON: %v (%s)", err, buf.String())
	}
	if decoded.Prompt != "a\x1bb" {
		t.Fatalf("unexpected prompt round trip: %q", decoded.Prompt)
	}
}
