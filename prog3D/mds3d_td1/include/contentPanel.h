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

    virtual void draw(NVGcontext *ctx) override;
    // virtual bool mouseMotionEvent(const Vector2i& p, const Vector2i& rel, int button, int modifiers);
};

#endif // CONTENTPANEL_H
