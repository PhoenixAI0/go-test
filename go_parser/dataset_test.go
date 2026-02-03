package main

import (
	"bufio"
	"bytes"
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
	if !bytes.Contains([]byte(samples[0].Prompt), []byte("B: C3")) {
		t.Fatalf("expected prompt to include setup stone B: C3")
	}
}

func TestWriteSampleJSON(t *testing.T) {
	var buf bytes.Buffer
	writer := bufio.NewWriter(&buf)
	if err := writeSample(writer, Sample{Prompt: "board_size: 5\nmove:", Response: "A1", BoardSize: 5}); err != nil {
		t.Fatalf("writeSample: %v", err)
	}
	if err := writer.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	expected := "{\"prompt\": \"board_size: 5\\nmove:\", \"response\": \"A1\", \"board_size\": 5}\n"
	if buf.String() != expected {
		t.Fatalf("unexpected JSONL output: %s", buf.String())
	}
}
