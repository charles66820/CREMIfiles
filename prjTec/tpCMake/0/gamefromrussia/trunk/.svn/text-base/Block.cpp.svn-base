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
#include "Block.h"

// SDL includes
#include <SDL.h>

// Local includes
#include "BlockSurfaceManager.h"
#include "consts.h"

Block::Block()
: m_color(BlockEmpty)
{
	m_rect.x = 0;
	m_rect.y = 0;
	m_rect.w = BlockWidth;
	m_rect.h = BlockHeight;
}

Block::Block(BlockColor color)
: m_color(color)
{
	m_rect.x = 0;
	m_rect.y = 0;
	m_rect.w = BlockWidth;
	m_rect.h = BlockHeight;
}

Block::~Block(void)
{
}

Block::Block(const Block &copy)
{
	m_color = copy.m_color;
	m_rect.x = copy.m_rect.x;
	m_rect.y = copy.m_rect.y;
	m_rect.w = copy.m_rect.w;
	m_rect.h = copy.m_rect.h;
}

Block &Block::operator=(const Block &other)
{
	if( this != &other )
	{
		m_color = other.m_color;
		m_rect.x = other.m_rect.x;
		m_rect.y = other.m_rect.y;
		m_rect.w = other.m_rect.w;
		m_rect.h = other.m_rect.h;
	}

	return *this;
}

void Block::setX(Sint16 value)
{
	m_rect.x = value;
}

void Block::setY(Sint16 value)
{
	m_rect.y = value;
}

void Block::moveAbsolute(Sint16 x, Sint16 y)
{
	m_rect.x = x;
	m_rect.y = y;
}

void Block::move(Sint16 deltaX, Sint16 deltaY)
{
	m_rect.x += deltaX;
	m_rect.y += deltaY;
}

inline SDL_Rect *Block::boundingRect()
{
	return &m_rect;
}

inline SDL_Surface *Block::getSurface()
{
	return BlockSurfaceManager::get( color() );
}

void Block::render(SDL_Surface *surface)
{
	render(surface, boundingRect());
}

void Block::render(SDL_Surface *surface, SDL_Rect *rect)
{
	SDL_BlitSurface(getSurface(), NULL, surface, rect);
}
