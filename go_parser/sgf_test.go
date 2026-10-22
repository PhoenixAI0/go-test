package main

import (
	"os"
	"testing"
)

func writeTempSGF(t *testing.T, content string) string {
	t.Helper()
	file, err := os.CreateTemp("", "sgf_*.sgf")
	if err != nil {
		t.Fatalf("create temp sgf: %v", err)
	}
	if _, err := file.WriteString(content); err != nil {
		file.Close()
		os.Remove(file.Name())
		t.Fatalf("write temp sgf: %v", err)
	}
	if err := file.Close(); err != nil {
		os.Remove(file.Name())
		t.Fatalf("close temp sgf: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(file.Name()) })
	return file.Name()
}

func TestParseSGFSetupStonesRange(t *testing.T) {
	path := writeTempSGF(t, "(;GM[1]FF[4]SZ[5]AB[aa:ab]AW[cc];B[dd])")
	game, err := parseSGFFile(path)
	if err != nil {
		t.Fatalf("parse SGF: %v", err)
	}
	if game.BoardSize != 5 {
		t.Fatalf("expected board size 5, got %d", game.BoardSize)
	}
	if len(game.SetupStones) != 3 {
		t.Fatalf("expected 3 setup stones, got %d", len(game.SetupStones))
	}
	if len(game.Moves) != 1 {
		t.Fatalf("expected 1 move, got %d", len(game.Moves))
	}
}

func TestParseSGFVariationsMainLine(t *testing.T) {
	path := writeTempSGF(t, "(;GM[1]FF[4]SZ[5];B[aa](;W[bb];B[cc])(;W[dd];B[ee]))")
	game, err := parseSGFFile(path)
	if err != nil {
		t.Fatalf("parse SGF: %v", err)
	}
	if len(game.Moves) != 3 {
		t.Fatalf("expected 3 moves, got %d", len(game.Moves))
	}
	if game.Moves[1].Player != "W" || game.Moves[1].Coords == nil {
		t.Fatalf("expected second move to be W with coords")
	}
}

func TestParseSGFF3PassMove(t *testing.T) {
	path := writeTempSGF(t, "(;GM[1]FF[3]SZ[19];B[dd];W[tt];B[pp])")
	game, err := parseSGFFile(path)
	if err != nil {
		t.Fatalf("parse SGF: %v", err)
	}
	if len(game.Moves) != 3 {
		t.Fatalf("expected 3 moves, got %d", len(game.Moves))
	}
	if game.Moves[1].Coords != nil {
		t.Fatalf("expected FF[3] tt move to be PASS")
	}
}

func TestParseSGFFF4TtPassMove(t *testing.T) {
	// Many FF[4] writers still emit FF[3]-style [tt] passes on small boards.
	path := writeTempSGF(t, "(;GM[1]FF[4]SZ[9];B[dd];W[tt];B[ee])")
	game, err := parseSGFFile(path)
	if err != nil {
		t.Fatalf("parse SGF: %v", err)
	}
	if len(game.Moves) != 3 || game.Moves[1].Coords != nil {
		t.Fatalf("expected FF[4] tt move to be PASS, got %+v", game.Moves)
	}
}

func TestParseSGFRejectsInvalidBoardSize(t *testing.T) {
	for _, size := range []string{"-1", "0", "25", "x"} {
		path := writeTempSGF(t, "(;GM[1]FF[4]SZ["+size+"];B[aa])")
		if _, err := parseSGFFile(path); err == nil {
			t.Fatalf("expected SZ[%s] to be rejected", size)
		}
	}
}

func TestParseSGFRejectsUnsupportedStateChanges(t *testing.T) {
	for _, content := range []string{
		"(;GM[2]SZ[5];B[aa])", "(;SZ[5]KM[NaN];B[aa])",
		"(;SZ[5];B[aa];AB[bb];W[cc])", "(;SZ[5];B[aa]W[bb])",
		"(;SZ[5]PL[Z];B[aa])", "(;SZ[5];B[aa];PL[B];B[bb])",
	} {
		if _, err := parseSGFFile(writeTempSGF(t, content)); err == nil {
			t.Fatalf("accepted invalid state: %s", content)
		}
	}
}

func TestParseSGFRootMoveAndMissingKomi(t *testing.T) {
	game, err := parseSGFFile(writeTempSGF(t, "(;SZ[5]B[aa];W[bb])"))
	if err != nil || len(game.Moves) != 2 || game.Komi != nil {
		t.Fatalf("root move lost or komi fabricated: %+v %v", game, err)
	}
}
