using Tobii.XR;
using UnityEngine;
using UnityEngine.UI;

public class gazePosition : MonoBehaviour
{
    Text text;

    // Start is called before the first frame update
    void Start()
    {
        text = gameObject.GetComponent<Text>();
    }

    // Update is called once per frame
    void Update()
    {
        var eyeTrackingData = TobiiXR.GetEyeTrackingData(TobiiXR_TrackingSpace.World);
        text.text = eyeTrackingData.GazeRay.Direction.ToString();

    }
}
