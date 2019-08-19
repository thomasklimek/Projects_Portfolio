package pacman.controllers;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

/*
 * A simple key adapter used by the HumanController to play the game.
 */
public class KeyBoardInput extends KeyAdapter
{
    private int key;
    private boolean newKey = false;

    public int getKey()
    {
    	if (newKey) {
    		newKey = false;
    		return key;
    		
    	}
    	else {
    		return 0;
    	}
    }

    public void keyPressed(KeyEvent e) 
    {
    	newKey = true;
        key=e.getKeyCode();
    }
}