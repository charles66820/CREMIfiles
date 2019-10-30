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
#include "DropPieceState.h"

#include <cstdlib>

DropPieceState::DropPieceState(GameStateMachine *machine)
: GameState(machine), dropDownNow(0), dropDownLastTime(0)
{
}

DropPieceState::~DropPieceState(void)
{
}

void DropPieceState::enter()
{
}

void DropPieceState::execute()
{
	// Gets a local reference to not call
	// data() too much.
	// Unless it crash randomly
	GameData* data = this->data();

	// Handle events
	while( SDL_PollEvent(&data->event) )
	{
		switch(data->event.type)
		{
			case SDL_KEYUP:
			{
				if( data->event.key.keysym.sym == SDLK_ESCAPE )
				{
					::exit(0);
				}
				if( data->event.key.keysym.sym == SDLK_UP )
				{
					data->currentPiece->rotate();
					data->player.playSound(SoundRotate);
				}
				if( data->event.key.keysym.sym == SDLK_LEFT )
				{
					data->currentPiece->stepLeft();
					data->player.playSound(SoundMove);
				}
				if( data->event.key.keysym.sym == SDLK_RIGHT )
				{
					data->currentPiece->stepRight();
					data->player.playSound(SoundMove);
				}
				if( data->event.key.keysym.sym == SDLK_RETURN )
				{
					stateMachine()->setNextState(PauseStateId);
					// TODO: Play pause sound
					return;
				}
				break;
			}
			case SDL_KEYDOWN:
				if( data->event.key.keysym.sym == SDLK_DOWN )
				{
					if( data->currentPiece->dropDown() )
					{
						// Add piece's block to the board
						data->board.addBlocks(data->currentPiece);
						delete data->currentPiece;
						data->currentPiece = 0;

						// Increase score
						data->score += rand() % 10;
						data->printer.setValue(TextScore, data->score);

						// Play drop sound
						data->player.playSound(SoundDrop);

						// Remove lines from board
						stateMachine()->setNextState(LinesRemoveAnimationStateId);
						return;
					}
				}
				break;
			case SDL_QUIT:
				::exit(0);
		}
	}

	// Render all graphics elements
	SDL_BlitSurface(data->background, NULL, data->gameSurface, NULL);

	data->board.render(data->gameSurface);
	if( data->currentPiece )
	{
		data->currentPiece->render(data->gameSurface);
	}
	if( data->nextPiece )
	{
		data->nextPiece->render(data->gameSurface);
	}

	// Render text last
	data->printer.render( data->gameSurface );

	dropDownNow = data->ticks;
	// Drop Down management
	if( dropDownNow - dropDownLastTime > data->dropDownDelay )
	{
		if( data->currentPiece->dropDown() )
		{
			// Add piece's block to the board
			data->board.addBlocks(data->currentPiece);
			delete data->currentPiece;
			data->currentPiece = 0;

			// Increase score
			data->score += rand() % 10;
			data->printer.setValue(TextScore, data->score);

			// Play drop sound
			data->player.playSound(SoundDrop);

			// Remove lines from board
			stateMachine()->setNextState(LinesRemoveAnimationStateId);
		}

		dropDownLastTime = dropDownNow;
	}
}

void DropPieceState::exit()
{
}