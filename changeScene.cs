using System;
using System.Collections;
using System.Collections.Generic;
using Tobii.G2OM;
using Tobii.XR;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class changeScene : MonoBehaviour, IGazeFocusable
{
    public GameObject progressBar;
    public GameObject vica;

    public void GazeFocusChanged(bool hasFocus)
    {
        if (hasFocus)
        {
            this.GetComponent<Image>().color = Color.green;
            progressBar.SetActive(true);
            vica.GetComponent<loadingbar>().SetFlagFocus(true);
        }
        else
        {
            this.GetComponent<Image>().color = Color.cyan;
            progressBar.SetActive(false);
            vica.GetComponent<loadingbar>().SetFlagFocus(false);
            vica.GetComponent<loadingbar>().ResetFillAmount();
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        var eyeTrackingData = TobiiXR.GetEyeTrackingData(TobiiXR_TrackingSpace.World);
       
    }

}
