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

#include "GameState.h"

struct LineRemoveData;

class LinesRemoveAnimationState : public GameState
{
public:
	LinesRemoveAnimationState(GameStateMachine *machine);
	~LinesRemoveAnimationState();

	void enter();
	void execute();
	void exit();

private:
	// Use a static array instead of std::vector
	// to reduce functions calls and dereferencement
	// of iterators. Unless it cause random crashes sometimes
	// for an unknown reason
	LineRemoveData* linesMask[4];
	int m_linesToRemove;

	// The separation x between erased and not
	int currentX;
	// Declared here to not been created on each execute() call
	SDL_Rect maskRect;
	SDL_Rect sourceRect;
	
};
