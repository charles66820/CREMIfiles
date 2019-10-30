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
#include "BlockSurfaceManager.h"

// STL includes
#include <map>

// Local includes
#include "consts.h"

BlockSurfaceManager BlockSurfaceManager::s_self;

#define d_ptr s_self.d

class BlockSurfaceManager::Private
{
public:
	~Private()
	{
		std::map<BlockColor, SDL_Surface*>::iterator it, itEnd = surfaceCache.end();
		for(it = surfaceCache.begin(); it != itEnd; ++it)
		{
			SDL_FreeSurface(it->second);
		}
	}
	std::map<BlockColor, SDL_Surface*> surfaceCache;
};

BlockSurfaceManager::BlockSurfaceManager()
: d(new Private)
{
}

BlockSurfaceManager::~BlockSurfaceManager()
{
	delete d;
}

SDL_Surface *BlockSurfaceManager::get(BlockColor color)
{
	if( hasNoCachedSurfaceFor(color) )
	{
		createCachedSurfaceFor(color);
	}

	return d_ptr->surfaceCache[color];
}

bool BlockSurfaceManager::hasNoCachedSurfaceFor(BlockColor &color)
{
	return d_ptr->surfaceCache.count(color) == 0;
}

void BlockSurfaceManager::createCachedSurfaceFor(BlockColor &color)
{
	d_ptr->surfaceCache[color] = createBlockSurface(color);
}

SDL_Surface* BlockSurfaceManager::createBlockSurface(BlockColor &color)
{
	SDL_Surface *newSurface = SDL_CreateRGBSurface(SDL_HWSURFACE, BlockWidth, BlockHeight, ColorDepth, 0, 0, 0, 0);

	Uint8 r=0,g=0,b=0;

	getBlockColorRgb(color, r, g, b);

	drawBlockSurface(newSurface, r, g, b);

	return newSurface;
}

void BlockSurfaceManager::drawBlockSurface(SDL_Surface *newSurface, Uint8 &r, Uint8 &g, Uint8 &b)
{
	SDL_Rect blockRect = createBlockRect();
	SDL_FillRect(newSurface, &blockRect, SDL_MapRGB(newSurface->format, r,g,b));
}

SDL_Rect BlockSurfaceManager::createBlockRect()
{
	SDL_Rect blockRect;
	blockRect.x=0;
	blockRect.y=0;
	blockRect.w=BlockWidth-1;
	blockRect.h=BlockHeight-1;
	return blockRect;
}

void BlockSurfaceManager::getBlockColorRgb(BlockColor &color, Uint8 &r, Uint8 &g, Uint8 &b)
{
	switch(color)
	{
	case BlockCyan:
		r=0; g=240; b=240;
		break;
	case BlockBlue:
		r=0; g=0; b=240;
		break;
	case BlockOrange:
		r=240; g=160; b=0;
		break;
	case BlockYellow:
		r=240; g=240; b=0;
		break;
	case BlockGreen:
		r=0; g=240; b=0;
		break;
	case BlockPurple:
		r=160; g=0; b=240;
		break;
	case BlockRed:
		r=240; g=0; b=0;
		break;
	}
}
