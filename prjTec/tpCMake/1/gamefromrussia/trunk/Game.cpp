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
#include "Game.h"

// STL includes
#include <iostream>
#include <cstdlib>

#include "GameData.h"

// SDL includes
#include <SDL_image.h>

// Local includes
#include "Block.h"
#include "BlockLine.h"
#include "PieceFactory.h"
#include "GameStateMachine.h"

using namespace std;


Game::Game()
: d(new GameData)
{
	stateMachine = new GameStateMachine(this);

	initSDL();

	stateMachine->setNextState(TitleScreenStateId);
}

Game::~Game()
{
	delete d;
}

void Game::initSDL()
{
	atexit(SDL_Quit);

	if( SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) == -1 )
	{
		cerr << "Couldn't initialise SDL. Error: " << SDL_GetError() << endl;
		exit(1);
	}

	SDL_WM_SetCaption("That game from Russia (you know it!)", NULL);
	d->screen = SDL_SetVideoMode(WindowWidth, WindowHeight, ColorDepth, SDL_SWSURFACE | SDL_DOUBLEBUF);
	if( !d->screen )
	{
		cerr << "Couldn't set " << WindowWidth << "x" << WindowHeight << " " << ColorDepth << "bit video mode." << endl;
		exit(1);
	}

	SDL_EnableKeyRepeat(10, 10);
	d->background = IMG_Load(DATADIR"background.png");

	// Create game surface (for alpha blending popup such as game over and pause)
	SDL_Surface *temp = SDL_CreateRGBSurface(SDL_HWSURFACE | SDL_SRCALPHA, WindowWidth, WindowHeight, ColorDepth, 0,0,0,0);
	d->gameSurface = SDL_DisplayFormat( temp );
	SDL_FreeSurface( temp );

	// Init text subsystem
	d->printer.init();

	// Init audio subsystem
	d->player.init();
}

bool Game::nextPiece()
{
	int tryOut=0;

	d->currentPiece = d->nextPiece;

	int y=BoardY;
	d->currentPiece->move(BoardX+(BlockWidth*5), y);
	while( !d->currentPiece->tryMove(0, y) && tryOut <= 4 )
	{
		y+=BlockHeight;
		tryOut++;
	}

	// Game is over folks
	if( tryOut >= 4 )
	{
		return false;
	}

	d->currentPiece->move(BoardX+(BlockWidth*5), y);

	d->nextPiece = PieceFactory::self()->randomPiece(&d->board);
	d->nextPiece->move(NextPieceX, NextPieceY);

	return true;
}

int Game::levelDropDown(int level)
{
	int result = -40 * (level-1) + DebutDropDownTime;

	if(result < MinimumDropDownTime)
	{
		result = MinimumDropDownTime;
	}

	return result;
}

bool Game::run()
{
	int frameRateNow=0, framerateLastTime=0;
	
	// Game loop
	while(1)
	{
		// Clear the screen, unless alpha blending doesn't work
		SDL_FillRect( d->screen, 0, SDL_MapRGBA(d->gameSurface->format, 0, 0, 0, 0) );

		// Blit game surface
		SDL_BlitSurface(d->gameSurface, NULL, d->screen, NULL);

		// Get current ticks
		d->ticks = SDL_GetTicks();

		// Execute current state
		stateMachine->execute();

		// Framerate management
		frameRateNow = d->ticks;
		if (frameRateNow - framerateLastTime > FrameRate)
		{
			framerateLastTime = frameRateNow;
		}
		else
		{
			SDL_Delay(FrameRate - (frameRateNow-framerateLastTime));
		}

		SDL_Flip(d->screen);
	}

	return true;
}
