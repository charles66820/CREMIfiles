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
#include "TextPrinter.h"

// STL includes
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>
#include <iostream>
#include <cctype>

// SDL includes
#include <SDL.h>
#include <SDL_ttf.h>

// Local includes
#include "consts.h"

using namespace std;

struct TextPrinterCache
{
	TextPrinterCache()
		: textSurface(0)
	{
	}
	SDL_Surface *textSurface;
	SDL_Rect textRect;
};

class TextPrinter::Private
{
public:
	Private()
	 : font(0)
	{
	}

	~Private()
	{
		map<TextCategory, TextPrinterCache>::iterator it, itEnd = cache.end();
		for(it = cache.begin(); it != itEnd; ++it)
		{
			if( it->second.textSurface )
			{
				SDL_FreeSurface( it->second.textSurface );
			}
		}

		TTF_CloseFont(font);
	}

	map<TextCategory, TextPrinterCache> cache;
	TTF_Font *font;
};

TextPrinter::TextPrinter()
: d(new Private)
{
}

TextPrinter::~TextPrinter(void)
{
	delete d;
}

void TextPrinter::init()
{
	atexit(TTF_Quit);

	if( TTF_Init() == -1 )
	{
		cerr << "Error while initiating font subsystem: " << TTF_GetError() << endl;
		exit(1);
	}

	d->font = TTF_OpenFont(DATADIR"nrkis.ttf", FontSize);
	if( !d->font )
	{
		cerr << "Can't open font nrkis.ttf: " << TTF_GetError() << endl;
		exit(1);
	}

	TextPrinterCache levelCache;
	levelCache.textRect = createRect(80,38);
	levelCache.textSurface = renderText( "01" );

	TextPrinterCache scoreCache;
	scoreCache.textRect = createRect(460, 36);
	scoreCache.textSurface = renderText( "000000000" );

	TextPrinterCache linesCache;
	linesCache.textRect = createRect(470, 310);
	linesCache.textSurface = renderText( "000000" );

	d->cache[TextLevel] = levelCache;
	d->cache[TextScore] = scoreCache;
	d->cache[TextLines] = linesCache;
}

void TextPrinter::render(SDL_Surface *surface)
{
	map<TextCategory, TextPrinterCache>::iterator it, itEnd = d->cache.end();
	for(it = d->cache.begin(); it != itEnd; ++it)
	{
		if( it->second.textSurface )
		{
			SDL_BlitSurface( it->second.textSurface, NULL, surface, &it->second.textRect);
		}
	}
}

void TextPrinter::setValue(TextCategory category, int value)
{
	stringstream stream;
	// Find number of digits
	int numberDigits=0, temp = value;
	while( temp != 0 )
	{
		temp /= 10;
		numberDigits++;
	}
	int numberLength=0;
	switch(category)
	{
		case TextLevel:
			numberLength = 2;
			break;
		case TextLines:
			numberLength = 6;
			break;
		case TextScore:
			numberLength = 9;
			break;
	}
	// Pad '0'
	for(int i=0; i<numberLength-numberDigits; i++)
	{
		stream << '0';
	}
	// Do not pad unnecessary 0
	if( value != 0 )
	{
		stream << value;
	}
	string text = stream.str();

	SDL_FreeSurface(d->cache[category].textSurface);
	d->cache[category].textSurface = renderText( text.c_str() );
}

SDL_Surface *TextPrinter::renderText(const char *text)
{
	static SDL_Color white = {255,255,255};
	SDL_Surface *renderedText = TTF_RenderText_Solid(d->font, text, white);

	return renderedText;
}

SDL_Rect TextPrinter::createRect(int x, int y)
{
	SDL_Rect rect;
	rect.x = x;
	rect.y = y;
	
	return rect;
}