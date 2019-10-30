// That Game From Russia
// Copyright (c) 2009 Michaël Larouche <larouche@kde.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#pragma once

// SDL includes
#include <SDL.h>

// Local includes
#include "Board.h"
#include "consts.h"
#include "Piece.h"
#include "TextPrinter.h"
#include "SoundPlayer.h"

// TODO: Move variables only applicable to Game
class GameData
{
public:
	GameData()
		: screen(0), background(0), gameSurface(0), currentPiece(0), nextPiece(0),
		  dropDownDelay(DebutDropDownTime), lines(0), level(1), score(0), ticks(0),
		  selectedMusic(MusicNewSchool), selectedLevel(1)
	{}
	~GameData()
	{
		delete currentPiece;
		delete nextPiece;
		SDL_FreeSurface(background);
		SDL_FreeSurface(gameSurface);
	}

	// SDL
	SDL_Surface *screen;
	SDL_Surface *background;
	SDL_Surface *gameSurface;
	SDL_Event event;

	// Board and pieces
	Board board;
	Piece *currentPiece;
	Piece *nextPiece;

	// Gameplay
	int dropDownDelay;
	int lines;
	int level;
	int score;

	// Miscs
	TextPrinter printer;
	SoundPlayer player;
	int ticks;

	// Setup
	Music selectedMusic;
	int selectedLevel;
};
