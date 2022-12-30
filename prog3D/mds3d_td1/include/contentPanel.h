#ifndef CONTENTPANEL_H
#define CONTENTPANEL_H

#include "block.h"
#include "scene.h"

#include <nanogui/glutil.h>
#include <nanogui/screen.h>

class ContentPanel : public nanogui::Widget
{
    nanogui::Screen* m_screen = nullptr;

protected:
    /** This method is automatically called everytime the mouse pointer change */
    // virtual bool mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers);

public:
    // default constructor
    ContentPanel(nanogui::Screen* screen);
    ~ContentPanel();

    // virtual bool mouseButtonEvent(const Vector2i& p, int button, bool down, int modifiers);
    virtual bool mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers);
    // virtual bool mouseDragEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers);
    // virtual bool mouseEnterEvent(const Vector2i& p, bool enter);
};

#endif // CONTENTPANEL_H
