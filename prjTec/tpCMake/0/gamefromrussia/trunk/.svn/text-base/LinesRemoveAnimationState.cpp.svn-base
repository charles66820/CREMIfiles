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
#include "LinesRemoveAnimationState.h"

// STL includes
#include <list>

struct LineRemoveData
{
	~LineRemoveData()
	{
		SDL_FreeSurface( maskSurface );
	}
	SDL_Surface *maskSurface;
	SDL_Rect position;
};

LinesRemoveAnimationState::LinesRemoveAnimationState(GameStateMachine *machine)
: GameState(machine), m_linesToRemove(0), currentX(0)
{
}

LinesRemoveAnimationState::~LinesRemoveAnimationState()
{
	for(int i=0; i<m_linesToRemove; i++)
	{
		delete linesMask[i];
	}
}

void LinesRemoveAnimationState::enter()
{
	std::vector<int> linesPosition = data()->board.fullLinesPosition();

	// Keep line count for static array
	m_linesToRemove = linesPosition.size();

	// Index for static LineRemoveData Array
	int i=0;

	// Create each mask surface for each line
	std::vector<int>::const_iterator it, itEnd = linesPosition.end();
	for(it = linesPosition.begin(); it != itEnd; ++it)
	{
		SDL_Surface *tempSurface = SDL_CreateRGBSurface(SDL_HWSURFACE | SDL_SRCALPHA, BoardRealWidth, BlockHeight, ColorDepth, 0,0,0,0);
		SDL_Surface *maskSurface = SDL_DisplayFormatAlpha(tempSurface);

		LineRemoveData *data = new LineRemoveData;
		data->position.x = BoardX;
		data->position.y = *it;
		data->position.h = BlockHeight;
		data->position.w = BoardWidth*BlockWidth;
		data->maskSurface = maskSurface;

		//linesMask.push_back( data );
		linesMask[i] = data;
		i++;
	}

	if( m_linesToRemove == 4 )
	{
		data()->player.playSound(SoundTetris);
	}
	else if( m_linesToRemove > 0 )
	{
		data()->player.playSound(SoundLineRemove);
	}
}

void LinesRemoveAnimationState::execute()
{
	// Do nothing if we don't have any lines to remove
	if( m_linesToRemove == 0 )
	{
		stateMachine()->setNextState(RemoveLinesStateId);
		return;
	}

	if( currentX <= BoardRealWidth )
	{
		for(int i=0; i<m_linesToRemove; i++)
		{
			LineRemoveData *lineData = linesMask[i];
			maskRect.y = 0;
			maskRect.h = lineData->position.h;
			maskRect.w = currentX;
			maskRect.x = 0;

			sourceRect.x = lineData->position.x;
			sourceRect.y = lineData->position.y;
			sourceRect.w = currentX;
			sourceRect.h = BlockHeight;

			// Fill rect up to currentX with background
			SDL_BlitSurface(data()->background, &sourceRect, lineData->maskSurface, &maskRect);

			// Fill rect from currentX with game surface
			maskRect.x = currentX;
			maskRect.w = lineData->position.w - currentX;
			sourceRect.x = sourceRect.x + currentX;
			sourceRect.w = lineData->position.w - currentX;

			SDL_BlitSurface(data()->gameSurface, &sourceRect, lineData->maskSurface, &maskRect);

			// Blit the surface
			SDL_BlitSurface(lineData->maskSurface, NULL, data()->gameSurface, &lineData->position);
		}
		currentX += BlockWidth;
	}
	else
	{
		stateMachine()->setNextState(RemoveLinesStateId);
	}
}

void LinesRemoveAnimationState::exit()
{
}
