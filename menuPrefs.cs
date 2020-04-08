using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class menuPrefs
{
    private static int volume;
    private static bool music;

    public static int Volume
    {
        get
        {
            return volume;
        }
        set
        {
            volume = value;
        }
    }

    public static bool Music
    {
        get
        {
            return music;
        }
        set
        {
            music = value;
        }
    }


}
