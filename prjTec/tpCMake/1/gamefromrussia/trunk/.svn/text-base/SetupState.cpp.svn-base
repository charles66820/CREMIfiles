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
#include "SetupState.h"

// SDL includes
#include <SDL_image.h>

// Local includes
#include "Cursor.h"

const int CursorDeltaX = 25;
const int CursorDeltaY = 45;

SetupState::SetupState(GameStateMachine *machine)
 : GameState(machine), m_selectedLevel(1), m_selectedMusic(MusicNewSchool)
{
	m_levelCursor = new Cursor(DirectionUp);
	m_musicCursor = new Cursor(DirectionLeft);
}

SetupState::~SetupState()
{
	SDL_FreeSurface(m_setupImage);
	delete m_levelCursor;
	delete m_musicCursor;
}

void SetupState::enter()
{
	// Load setup background image
	m_setupImage = IMG_Load(DATADIR"setup.png");

	// Load first music in selection
	data()->player.changeMusic( static_cast<Music>(m_selectedMusic) );

	// Move cursor to correct position
	m_levelCursor->move(185, 115);
	m_musicCursor->move(170, 240);
}

void SetupState::execute()
{
	while( SDL_PollEvent(&data()->event) )
	{
		switch( data()->event.type )
		{
			case SDL_KEYUP:
			{
				if( data()->event.key.keysym.sym == SDLK_RETURN || data()->event.key.keysym.sym == SDLK_SPACE )
				{
					stateMachine()->setNextState(InitGameStateId);
					return;
				}
				if( data()->event.key.keysym.sym == SDLK_ESCAPE )
				{
					::exit(0);
				}
				if( data()->event.key.keysym.sym == SDLK_LEFT )
				{
					if( m_selectedLevel - 1 >= 1 )
					{
						m_selectedLevel--;
						m_levelCursor->move(-CursorDeltaX, 0);
					}
				}
				if( data()->event.key.keysym.sym == SDLK_RIGHT )
				{
					if( m_selectedLevel + 1 <= 10 )
					{
						m_selectedLevel++;
						m_levelCursor->move(CursorDeltaX, 0);
					}
				}
				if( data()->event.key.keysym.sym == SDLK_UP )
				{
					if( m_selectedMusic - 1 >= MusicNewSchool )
					{
						m_selectedMusic--;
						m_musicCursor->move(0, -CursorDeltaY);
						data()->player.changeMusic( static_cast<Music>(m_selectedMusic) );
					}
				}
				if( data()->event.key.keysym.sym == SDLK_DOWN )
				{
					if( m_selectedMusic + 1 <= MusicNoMusic )
					{
						m_selectedMusic++;
						m_musicCursor->move(0, CursorDeltaY);
						data()->player.changeMusic( static_cast<Music>(m_selectedMusic) );
					}
				}
				break;
			}
			case SDL_QUIT:
			{
				::exit(0);
				break;
			}
		}
	}

	// Render graphics elements
	SDL_BlitSurface(m_setupImage, NULL, data()->gameSurface, NULL);
	m_levelCursor->render(data()->gameSurface);
	m_musicCursor->render(data()->gameSurface);
}

void SetupState::exit()
{
	data()->selectedLevel = m_selectedLevel;
	data()->selectedMusic = static_cast<Music>(m_selectedMusic);
}
