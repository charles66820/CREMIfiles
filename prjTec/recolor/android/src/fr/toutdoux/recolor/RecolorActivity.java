package fr.toutdoux.recolor;

import org.libsdl.app.SDLActivity;

import android.app.Activity;
import android.content.*;
import android.os.Bundle;
import android.util.Log;
import android.view.*;
import android.widget.CompoundButton;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;

public class RecolorActivity extends Activity {
    private static final String TAG = "RECOLOR";

    public static final String EXTRA_MESSAGE = "fr.toutdoux.recolor.MESSAGE";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // render main activity
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recolor);
    }

    public void sendMessage(View view) {
        Intent intent = new Intent(this, SDLActivity.class); // instantiate Intent with an new activity
        EditText editText = (EditText) findViewById(R.id.editTextArgs); // get text field
        String message = editText.getText().toString(); // get field content
        Log.d(TAG, "cc " + message);
        intent.putExtra(EXTRA_MESSAGE, message); // send msg to activity
        startActivity(intent);
    }
}