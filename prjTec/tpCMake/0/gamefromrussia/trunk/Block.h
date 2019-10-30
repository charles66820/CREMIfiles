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

#include "enums.h"

#include <SDL.h>

class Block
{
public:
	explicit Block();
	Block(BlockColor color);
	~Block();
	Block(const Block &copy);

	Block &operator=(const Block &other);

	Sint16 x() const;

	void setX(Sint16 value);

	Sint16 y() const;

	void setY(Sint16 value);

	void moveAbsolute(Sint16 x, Sint16 y);

	void move(Sint16 deltaX, Sint16 deltaY);

	// For everything else
	void render(SDL_Surface *surface);

	// For BlockLine
	void render(SDL_Surface *surface, SDL_Rect *rect);

	BlockColor color() const;

private:
	SDL_Rect *boundingRect();
	SDL_Surface *getSurface();

private:
	BlockColor m_color;
	SDL_Rect m_rect;
};

// Inline functions, for reducing functions calls
inline Sint16 Block::x() const
{
	return m_rect.x;
}

inline Sint16 Block::y() const
{
	return m_rect.y;
}

inline BlockColor Block::color() const
{
	return m_color;
}
