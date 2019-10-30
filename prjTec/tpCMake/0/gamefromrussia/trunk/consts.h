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

// Block height and width
static const int BlockHeight = 16;
static const int BlockWidth = 16;

// Board Height and Width (in Block size)
static const int BoardHeight = 20;
static const int BoardWidth = 10;
// Board Height and Width (in pixel size)
static const int BoardRealWidth = BoardWidth*BlockWidth;
static const int BoardRealHeight = BoardHeight*BlockHeight;
// Board position (used for collision detection)
static const int BoardX = 215;
static const int BoardY = 0;
static const int BoardEndX = BoardX+BoardWidth*BlockWidth;
static const int BoardEndY = BoardY+BoardHeight*BlockHeight;

// Window constants
static const int WindowWidth = 640;
static const int WindowHeight = 480;
static const int ColorDepth = 32;

// Number of blocks for a tetra, should never change :)
static const int TetraSize = 4;

// Framerate
static const int FrameRate = 60;

// Next piece position
static const int NextPieceX = BoardEndX+(8*BlockWidth);
static const int NextPieceY = 11*BlockHeight;

// Gameplay constants
static const int MaxLevel = 10;
static const int LinesPerLevel = 10;
static const int DebutDropDownTime = 500;
static const int MinimumDropDownTime = 100;

// Text constants
static const int FontSize = 24;

// Audio constants
static const int SampleRate = 44100;
static const int AudioChannels = 2;
static const int AudioBufferSize = 4096;
