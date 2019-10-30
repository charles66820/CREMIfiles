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
#include "TitleScreenState.h"

// SDL includes
#include <SDL_image.h>

TitleScreenState::TitleScreenState(GameStateMachine *machine)
: GameState(machine)
{
}

TitleScreenState::~TitleScreenState(void)
{
	SDL_FreeSurface(m_titlescreenImage);
}

void TitleScreenState::enter()
{
	m_titlescreenImage = IMG_Load(DATADIR"titlescreen.png");

	// Play title screen music
	data()->player.changeMusic(MusicTitleScreen);
}

void TitleScreenState::execute()
{
	while( SDL_PollEvent(&data()->event) )
	{
		switch( data()->event.type )
		{
			case SDL_KEYUP:
			{
				if( data()->event.key.keysym.sym == SDLK_RETURN || data()->event.key.keysym.sym == SDLK_SPACE )
				{
					stateMachine()->setNextState(SetupStateId);
					return;
				}
				if( data()->event.key.keysym.sym == SDLK_ESCAPE )
				{
					::exit(0);
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

	SDL_BlitSurface(m_titlescreenImage, NULL, data()->gameSurface, NULL);
}

void TitleScreenState::exit()
{
	// Stop title screen music
	data()->player.stopMusic();
}
