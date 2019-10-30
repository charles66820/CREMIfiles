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
#include "Cursor.h"

// SDL includes
#include <SDL.h>
#include <SDL_image.h>

class Cursor::Private
{
public:
	Private()
		: cursorImage(0)
	{
		position.x = 0;
		position.y = 0;
	}
	~Private()
	{
		SDL_FreeSurface(cursorImage);
	}

	Direction direction;
	SDL_Surface *cursorImage;
	SDL_Rect position;
};

Cursor::Cursor(Direction direction)
: d(new Private)
{
	d->direction = direction;

	if( direction == DirectionLeft )
	{
		d->cursorImage = IMG_Load(DATADIR"cursor_left.png");
	}
	else if( direction == DirectionUp )
	{
		d->cursorImage = IMG_Load(DATADIR"cursor_up.png");
	}
}

Cursor::~Cursor()
{
	delete d;
}

void Cursor::move(int deltaX, int deltaY)
{
	d->position.x += deltaX;
	d->position.y += deltaY;
}

void Cursor::render(SDL_Surface *surface)
{
	SDL_BlitSurface(d->cursorImage, NULL, surface, &d->position);
}
