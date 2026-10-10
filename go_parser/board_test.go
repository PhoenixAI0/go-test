package main

import "testing"

func TestCoordsRoundTrip(t *testing.T) {
	coords, err := gtpToCoords("C3", 5)
	if err != nil {
		t.Fatalf("gtpToCoords: %v", err)
	}
	gtp, err := coordsToGTP(coords, 5)
	if err != nil {
		t.Fatalf("coordsToGTP: %v", err)
	}
	if gtp != "C3" {
		t.Fatalf("expected C3, got %s", gtp)
	}
}

func TestApplyStrictMoveCapture(t *testing.T) {
	board := make([]byte, 9)
	// Surround center with three black stones, leave one liberty at (2,1).
	coords := []Coord{{Row: 0, Col: 1}, {Row: 1, Col: 0}, {Row: 1, Col: 2}}
	for _, c := range coords {
		var err error
		board, err = applyStrictMove(board, 3, stoneBlack, c)
		if err != nil {
			t.Fatalf("applyStrictMove black: %v", err)
		}
	}
	var err error
	board, err = applyStrictMove(board, 3, stoneWhite, Coord{Row: 1, Col: 1})
	if err != nil {
		t.Fatalf("applyStrictMove white: %v", err)
	}
	// Black plays at (2,1) to capture the white stone.
	board, err = applyStrictMove(board, 3, stoneBlack, Coord{Row: 2, Col: 1})
	if err != nil {
		t.Fatalf("applyStrictMove capture: %v", err)
	}
	if board[1*3+1] != stoneEmpty {
		t.Fatalf("expected center stone to be captured")
	}
}

func TestBoardSamplePositionsIllegalMoveStops(t *testing.T) {
	game := &SgfGame{
		BoardSize: 5,
		Moves: []Move{
			{Player: "B", Coords: &Coord{Row: 4, Col: 0}},
			{Player: "W", Coords: &Coord{Row: 4, Col: 0}},
		},
	}
	samples, err := samplePositions(game, 10)
	if err != nil {
		t.Fatalf("samplePositions: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("expected 1 legal sample, got %d", len(samples))
	}
	if samples[0].Response != "A1" {
		t.Fatalf("expected response A1, got %s", samples[0].Response)
	}
}
