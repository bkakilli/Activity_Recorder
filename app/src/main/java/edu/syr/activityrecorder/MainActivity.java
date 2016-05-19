package edu.syr.activityrecorder;

import android.app.Activity;
import android.hardware.SensorEventListener;
import android.os.Bundle;
import android.view.View;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Rect;
import org.opencv.core.TermCriteria;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint2f;
import org.opencv.video.Video;
import org.opencv.objdetect.HOGDescriptor;

import android.annotation.SuppressLint;
import android.app.Activity;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.ImageView;
import android.widget.Toast;

public class MainActivity extends Activity implements View.OnTouchListener, CvCameraViewListener2, SensorEventListener {

    private static final String  TAG                 = "OCVSample::Activity";

    public static final int      VIEW_MODE_RGBA       = 0;
    public static final int 	 VIEW_MODE_OPTICAL    = 4;
    public static final int      VIEW_MODE_HOG_RECORD = 5;
    public static final int 	 VIEW_MODE_RECORD     = 6;
    public static final int 	 VIEW_MODE_STOP		  = 7;

    static final int REQUEST_VIDEO_CAPTURE = 1;

    private static final int 	 MAX_CORNERS = 50;

    //private static final int     WINDOW_SIZE = 10;

    private SensorManager 		 mSensorManager;
    private Sensor 				 mSensor;
    private MenuItem             mItemPreviewRGBA;
    private MenuItem             mItemPreviewHist;
    private MenuItem             mItemPreviewCanny;
    private MenuItem             mItemPreviewAcc;
    private MenuItem             mItemPreviewHogs;
    private MenuItem             mItemPreviewAccHog;
    private MenuItem             mItemPreviewHog;
    private MenuItem 			 mItemPreviewOptical;
    private MenuItem			 mItemPreviewRecord;
    private MenuItem			 mItemPreviewStop;

    private CameraBridgeViewBase mOpenCvCameraView;

    private Mat                  mIntermediateMat;
    private Mat                  mMat0;
    private Mat 				 rgba;
    private Rect                 regionOfInterest;
    private Point                pointOfInterest;
    private MatOfInt             mChannels[];
    private MatOfInt             mHistSize;
    private int                  mHistSizeNum = 25;
    private int 				 mHistSizeStrength = 18; //18
    private int 				 mHistSizeOrientation= 9;
    private MatOfFloat           mRanges;
    private MatOfFloat           mStrength;
    private MatOfFloat			 mOrientation;
    private MatOfInt             mHistSizeS;
    private MatOfInt             mHistSizeO;
    private Scalar               mColorsRGB[];
    private Scalar               mColorsHue[];
    private Scalar               mWhilte;
    private Point                mP1;
    private Point                mP2;
    private Point 				 rectP1;
    private Point 				 rectP2;
    private float                mBuff[];
    private Mat                  mSepiaKernel;
    private int count = 0;
    private int countFrame =0;
    private int saveCount=0;
    private Mat prevRGB, nextRGB;
    private HOGDescriptor 		 hog;
    private Scalar               mBlobColorRgba;
    private Scalar               mBlobColorHsv;
    private ColorBlobDetector    mDetector;
    private Mat                  mSpectrum;
    private Size                 SPECTRUM_SIZE;
    private boolean              mIsColorSelected = false;
    private Scalar               BOUNDING_COLOR;
    private Scalar 				 CONTOUR_COLOR;
    private float 				 gravity[];
    private float 				 linear_acceleration[];
    private ImageView image;
    private int scale = 1;
    private int delta = 0;
    private int ddepth = CvType.CV_32F;
    double magnitude=0;
    double fallResult =0;
    private double fallMax = 0;
    private double magMax = 0;
    private double finalMax = 0;
    private boolean fallDetected = false;
    private boolean fallAccDetected = false;
    private boolean fallHogDetected = false;
    private Mat strengthNext;
    private Mat orientationNext;
    private int select = 0;
    private boolean writeComplete = false;
    private static final int VIDEO_CAPTURE = 101;
    FileWriter writer;

    File root = Environment.getExternalStorageDirectory();
    File gpxfile = new File(root, "mydata0.csv");

    public static int viewMode = VIEW_MODE_RGBA;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.findFocus();
                    //mOpenCvCameraView.setOnTouchListener(ImageManipulationsActivity.this);

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.image_manipulations_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setMaxFrameSize(350, 350);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        try {
            writer = new FileWriter(gpxfile);
            writeCsvHeader("Count","X","Y","Z","dEO","dES");
            //writeCsvHeader("Count","Acc","Cam","Acc+Cam","accVal", "camVal","fusionVal");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //Write CSV Header
    private void writeCsvHeader(String h1, String h2, String h3, String h4, String h5, String h6) throws IOException {
        String line = String.format("%s,%s,%s,%s,%s,%s\n", h1, h2, h3, h4, h5, h6);
        writer.write(line);
    }

    // Write CSV Data
    @SuppressLint("DefaultLocale")
    private void writeCsvData(int c, float x, float y, float z, double o, double s) throws IOException {
        String line = String.format("%d,%f,%f,%f,%f,%f\n", c, x, y, z, o, s);
        writer.write(line);
    }
    //Write CSV Header
    /*private void writeCsvHeader(String h1, String h2, String h3, String h4, String h5, String h6, String h7) throws IOException {
        String line = String.format("%s,%s,%s,%s,%s,%s,%s\n", h1, h2, h3, h4, h5, h6, h7);
        writer.write(line);
    }

    // Write CSV Data
    @SuppressLint("DefaultLocale")
    private void writeCsvData(int c, boolean d, boolean e, boolean f, double g, double h, double k) throws IOException {
        String line = String.format("%d,%b,%b,%b,%f,%f,%f\n", c, d, e, f, g, h, k);
        writer.write(line);
    }*/

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @Override
    public final void onSensorChanged(SensorEvent event) {
        // In this example, alpha is calculated as t / (t + dT),
        // where t is the low-pass filter's time-constant and
        // dT is the event delivery rate.

        final float alpha = (float) 0.9;
        // Isolate the force of gravity with the low-pass filter.
        gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0];
        gravity[1] = alpha * gravity[1] + (1 - alpha) * event.values[1];
        gravity[2] = alpha * gravity[2] + (1 - alpha) * event.values[2];

        // Remove the gravity contribution with the high-pass filter.
        linear_acceleration[0] = event.values[0] - gravity[0];
        linear_acceleration[1] = event.values[1] - gravity[1];
        linear_acceleration[2] = event.values[2] - gravity[2];
        magnitude = Math.sqrt(Math.pow(linear_acceleration[0], 2) + Math.pow(linear_acceleration[1], 2) + Math.pow(linear_acceleration[2], 2));
        magnitude /= Math.sqrt(3*Math.pow(mSensor.getMaximumRange(), 2));
        if (magnitude > magMax)
            magMax = magnitude;
        if (fallResult > fallMax)
            fallMax = fallResult;
        if (finalMax < magnitude+fallResult)
            finalMax = magnitude+fallResult;
        if (magnitude >= 0.5)
            fallAccDetected = true;
        if (fallResult >= 0.25)
            fallHogDetected = true;
        if (magnitude+fallResult > 0.7)
            fallDetected = true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        mSensorManager.unregisterListener(this);
    }

    @Override
    /* register the broadcast receiver with the intent values to be matched */

    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);
    }

    private boolean hasCamera() {
        if (getPackageManager().hasSystemFeature(
                PackageManager.FEATURE_CAMERA_ANY)){
            return true;
        } else {
            return false;
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void SaveImage (Mat mat, int frameNumber, int saveCount) {
        Mat mIntermediateMat = new Mat();
        Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_GRAY2BGRA, 3);

        File path = new File(Environment.getExternalStorageDirectory() + "/Images"+Integer.toString(saveCount)+"/");
        path.mkdirs();
        String fname="";

        if(frameNumber < 10){
            fname = "fall000"+ frameNumber +".jpg";
        }else if (frameNumber < 100){
            fname = "fall00"+ frameNumber +".jpg";
        }else if (frameNumber < 1000){
            fname = "fall0"+ frameNumber +".jpg";
        }else{
            fname = "fall"+ frameNumber +".jpg";
        }

        File file = new File(path, fname);

        String filename = file.toString();
        Boolean bool = Imgcodecs.imwrite(filename, mIntermediateMat);

        if (bool)
            Log.i(TAG, "SUCCESS writing image to external storage");
        else
            Log.i(TAG, "Fail writing image to external storage");
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA  = menu.add("Preview RGBA");
        //mItemPreviewHist  = menu.add("Histograms");
        //mItemPreviewCanny = menu.add("Canny");
        //mItemPreviewSepia = menu.add("Camera");
        //mItemPreviewOptical = menu.add("Optical Flow Tracker");
        mItemPreviewHog = menu.add("RecordHOG");
        mItemPreviewRecord = menu.add("RecordActivity");
        mItemPreviewStop = menu.add("Stop!");

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
            //else if (item == mItemPreviewOptical)
            //	viewMode = VIEW_MODE_OPTICAL;
        else if (item == mItemPreviewHog)
            viewMode = VIEW_MODE_HOG_RECORD;
        else if (item == mItemPreviewRecord)
            viewMode = VIEW_MODE_RECORD;
        else if (item == mItemPreviewStop)
            viewMode = VIEW_MODE_STOP;
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        rgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
        BOUNDING_COLOR = new Scalar(0,255,0,255);
        mIntermediateMat = new Mat();
        mChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mHistSizeS = new MatOfInt(mHistSizeStrength);
        mHistSizeO = new MatOfInt(mHistSizeOrientation);
        mRanges = new MatOfFloat(0f, 256f);
        mStrength = new MatOfFloat(0f,361f);
        mOrientation = new MatOfFloat(0f, 360f);
        mMat0  = new Mat();
        mColorsRGB = new Scalar[] { new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255) };
        mColorsHue = new Scalar[] {
                new Scalar(255, 0, 0, 255),   new Scalar(255, 60, 0, 255),  new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255),  new Scalar(20, 255, 0, 255),  new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255),  new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255),  new Scalar(0, 0, 255, 255),   new Scalar(64, 0, 255, 255),  new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255),  new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();
        prevRGB = new Mat();
        nextRGB = new Mat();
        strengthNext = new Mat();
        orientationNext = new Mat();
        //hog = new HOGDescriptor(new Size(320,240), new Size(1,1), new Size(0,0), new Size(4,4), 18);

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
        gravity = new float[3];
        linear_acceleration= new float[3];

        //mController = new CPUController();


    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();

        mIntermediateMat = null;
        rgba.release();
    }

    @SuppressLint("ClickableViewAccessibility")
    public boolean onTouch(View v, MotionEvent event) {
        int cols = rgba.cols();
        int rows = rgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;

        pointOfInterest = new Point(); // new point

        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        Rect touchedRect = new Rect();
        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

        // get the regionOfInterest
        regionOfInterest = touchedRect ;

        // get pointOfInterest
        pointOfInterest.x =  (double)x;
        pointOfInterest.y =  (double)y;

        // get color for touchedRegion
        Mat touchedRegionRgba = rgba.submat(touchedRect);
        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region

        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);

        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] +
                ", " + mBlobColorRgba.val[2] + ", " + mBlobColorRgba.val[3] + ")");

        mDetector.setHsvColor(mBlobColorHsv);

        Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

        mIsColorSelected = true;

        touchedRegionRgba.release();
        touchedRegionHsv.release();

        return false; // don't need subsequent touch events
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_VIDEO_CAPTURE && resultCode == RESULT_OK) {
            //Uri videoUri = intent.getData();
            //mVideoView.setVideoURI(videoUri);
        }
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);
        return new Scalar(pointMatRgba.get(0, 0));
    }

    public void copyFile(InputStream in, OutputStream out) throws IOException {

        // Transfer bytes from in to out
        byte[] buf = new byte[1024];
        int len;
        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
        in.close();
        out.close();
    }

    /*
    public void surfaceDestroyed(SurfaceHolder holder) {
        // Surface will be destroyed when we return, so stop the preview.
        if (mCamera != null) {
            // Call stopPreview() to stop updating the preview surface.
            mCamera.stopPreview();
        }
    }

    /**
     * When this function returns, mCamera will be null.
     */
    /*
    private void stopPreviewAndFreeCamera() {

        if (mCamera != null) {
            // Call stopPreview() to stop updating the preview surface.
            mCamera.stopPreview();

            // Important: Call release() to release the camera for use by other
            // applications. Applications should release the camera immediately
            // during onPause() and re-open() it during onResume()).
            mCamera.release();

            mCamera = null;
        }
    }*/

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        rgba = inputFrame.gray();
        Size sizeRgba = rgba.size();

        Mat rgbaInnerWindow, rgbaInnerWindow1, rgbaInnerWindow2;
        Mat cellWindow;
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        MatOfPoint2f nextPts = new MatOfPoint2f();
        MatOfPoint2f prevPts = new MatOfPoint2f();
        MatOfPoint initial = new MatOfPoint();

        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        int left = 2*cols/8;
        int top = 2*rows/8;

        int width = cols * 1/4;
        int height = rows * 1/4;

        if (mIsColorSelected) {
            // draw the touchedRegion
            Imgproc.rectangle(rgba, regionOfInterest.tl(), regionOfInterest.br(), CONTOUR_COLOR);

            // go to detection
            mDetector.process(rgba);

            // get contours
            List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());

            // convert list to array
            List<Rect> finalBox = new ArrayList<Rect>();
            Imgproc.drawContours(rgba, contours, -1, CONTOUR_COLOR);
            // get boundingBox for each contour
            for (int i = 0; i < contours.size(); i++) {
                finalBox.add(i, Imgproc.boundingRect(contours.get(i)));
                if(pointOfInterest.inside(finalBox.get(i))){
                    Imgproc.rectangle(rgba, finalBox.get(i).tl(), finalBox.get(i).br(), BOUNDING_COLOR,3);
                    rectP1 = finalBox.get(i).tl();
                    rectP2 = finalBox.get(i).br();
                }
            }

        }

        switch (MainActivity.viewMode) {
            case MainActivity.VIEW_MODE_RGBA:
                fallDetected = false;
                fallAccDetected = false;
                fallHogDetected = false;
                fallMax = 0;
                magMax = 0;
                finalMax=0;
                break;

        /*
        case ImageManipulationsActivity.VIEW_MODE_HIST:
            Mat hist = new Mat();
            int thikness = (int) (sizeRgba.width / (mHistSizeNum + 10) / 5);
            if(thikness > 5) thikness = 5;
            int offset = (int) ((sizeRgba.width - (5*mHistSizeNum + 4*10)*thikness)/2);
            // RGB
            for(int c=0; c<3; c++) {
                Imgproc.calcHist(Arrays.asList(rgba), mChannels[c], mMat0, hist, mHistSize, mRanges);
                Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
                hist.get(0, 0, mBuff);
                for(int h=0; h<mHistSizeNum; h++) {
                    mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness;
                    mP1.y = sizeRgba.height-1;
                    mP2.y = mP1.y - 2 - (int)mBuff[h];
                    Core.line(rgba, mP1, mP2, mColorsRGB[c], thikness);
                }
            }
            // Value and Hue
            Imgproc.cvtColor(rgba, mIntermediateMat, Imgproc.COLOR_RGB2HSV_FULL);
            // Value
            Imgproc.calcHist(Arrays.asList(mIntermediateMat), mChannels[2], mMat0, hist, mHistSize, mRanges);
            Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
            hist.get(0, 0, mBuff);
            for(int h=0; h<mHistSizeNum; h++) {
                mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness;
                mP1.y = sizeRgba.height-1;
                mP2.y = mP1.y - 2 - (int)mBuff[h];
                Core.line(rgba, mP1, mP2, mWhilte, thikness);
            }
            // Hue
            Imgproc.calcHist(Arrays.asList(mIntermediateMat), mChannels[0], mMat0, hist, mHistSize, mRanges);
            Core.normalize(hist, hist, sizeRgba.height/2, 0, Core.NORM_INF);
            hist.get(0, 0, mBuff);
            for(int h=0; h<mHistSizeNum; h++) {
                mP1.x = mP2.x = offset + (4 * (mHistSizeNum + 10) + h) * thikness;
                mP1.y = sizeRgba.height-1;
                mP2.y = mP1.y - 2 - (int)mBuff[h];
                Core.line(rgba, mP1, mP2, mColorsHue[h], thikness);
            }
            break;

        case ImageManipulationsActivity.VIEW_MODE_CANNY:
            rgbaInnerWindow = rgba.submat(top, top + height, left, left + width);
            Imgproc.Canny(rgbaInnerWindow, mIntermediateMat, 80, 90);
            Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
            rgbaInnerWindow.release();
            break;
        */

            case MainActivity.VIEW_MODE_STOP:
                fallDetected = false;
                if(writeComplete){
                    try {
                        writer.flush();
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                    try {
                        writer.close();
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                    countFrame = 0;
                    saveCount++;
                    String text = "mydata";
                    gpxfile = new File(root, text.concat(Integer.toString(saveCount).concat(".csv")));
                    try {
                        writer = new FileWriter(gpxfile);
                        //writeCsvHeader( "Count", "Acc", "Cam", "Acc+Cam", "accVal", "camVal", "fusionVal");
                        writeCsvHeader("Count","X","Y","Z","dEO","dES");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    writeComplete = false;
                }
                break;

            case MainActivity.VIEW_MODE_RECORD:
                //Record captured video activity
                fallDetected = false;
                writeComplete = true;
                rgbaInnerWindow = rgba.submat(0, rows, 0, cols);
                mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
                mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);

            /*
        	String accX = String.valueOf(linear_acceleration[0]);
        	String accY = String.valueOf(linear_acceleration[1]);
        	String accZ = String.valueOf(linear_acceleration[2]);
        	String textX = new String("X: ");
        	String textY = new String("Y: ");
        	String textZ = new String("Z: ");	*/

                //Core.putText(rgbaInnerWindow, textX.concat(accX), new Point(10,10), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,0,0,255));
                //Core.putText(rgbaInnerWindow, textY.concat(accY), new Point(10,30), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,200,0,255));
                //Core.putText(rgbaInnerWindow, textZ.concat(accZ), new Point(10,50), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,200,200,255));

            /*
            SaveImage(rgba, countFrame, saveCount);
        	try {
                writeCsvData(countFrame, linear_acceleration[0], linear_acceleration[1], linear_acceleration[2]);
        	} catch (IOException e) {
                e.printStackTrace();
        	}*/

        	/*
        	Intent takeVideoIntent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
            if (takeVideoIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takeVideoIntent, REQUEST_VIDEO_CAPTURE);
            }*/

                countFrame++;
                break;

            case MainActivity.VIEW_MODE_OPTICAL:

                rgba = inputFrame.rgba();
                nextRGB = inputFrame.gray();
                top = (int)rectP1.y;
                height =(int)(rectP2.y-rectP1.y);
                left = (int)(rectP1.x);
                width =(int)(rectP2.x-rectP1.x);

                Imgproc.rectangle(rgba, rectP1, rectP2, BOUNDING_COLOR,1);

                rgbaInnerWindow = rgba.submat(top, top + height, left, left + width);

                if (count > 1){

                    rgbaInnerWindow1 = nextRGB.submat(top, top + height, left, left + width);
                    rgbaInnerWindow2 = prevRGB.submat(top, top + height, left, left + width);

                    Mat next = new Mat(rgbaInnerWindow1.size(), CvType.CV_8UC1);
                    Mat prev = new Mat(rgbaInnerWindow2.size(), CvType.CV_8UC1);

                    TermCriteria optical_flow_termination_criteria = new TermCriteria(TermCriteria.MAX_ITER|TermCriteria.EPS,25,.3);// ( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 );
                    optical_flow_termination_criteria.epsilon  = .3;
                    optical_flow_termination_criteria.maxCount = 20;

                    // Good features to track
                    // Imgproc.cvtColor(prevRGB, prev, Imgproc.COLOR_RGBA2GRAY,0);
                    // Imgproc.cvtColor(rgbaInnerWindow, curr, Imgproc.COLOR_RGBA2GRAY,0);
                    // Imgproc.goodFeaturesToTrack(prevRGB, tmpCorners, 100, 0.001, 10.0);
                    // Video.calcOpticalFlowPyrLK(prev, curr,curCorners,tmpCorners,status,new MatOfFloat(), new Size(10,10), 3,optical_flow_termination_criteria, 0, 1);
                    //http://stackoverflow.com/questions/12561292/android-using-calcopticalflowpyrlk-with-matofpoint2f
                    //if(hp.previousCorners.total()>0 )
                    //Video.calcOpticalFlowPyrLK(prev, curr,hp.previousCorners,tmpCorners,status,new MatOfFloat(), new Size(11,11),5,optical_flow_termination_criteria, 0, 1);

                    Imgproc.goodFeaturesToTrack(rgbaInnerWindow2, initial, MAX_CORNERS, 0.01, 0.01);
                    initial.convertTo(prevPts, CvType.CV_32FC2);

                    Video.calcOpticalFlowPyrLK(prev, next, prevPts, nextPts, status, err, new Size(11,11),5,optical_flow_termination_criteria, 0, 1);
                    Point[] pointp = prevPts.toArray();
                    Point[] pointn = nextPts.toArray();
                    byte[] statArray = status.toArray();
                    for(Point px : pointn){
                        Imgproc.circle(rgbaInnerWindow, px, 5, new Scalar(200,200,200,255));
                        //if (px.x > rectP2.x){rectP2.x = px.x;}
                        //if (px.y > rectP2.y){rectP2.y = px.y;}
                        //if (px.x < rectP1.x){rectP1.x = px.x;}
                        //if (px.y < rectP1.y){rectP1.y = px.y;}

                    }

                    for (int i =0; i< pointn.length;i++)
                    {
                        if (statArray[i] == 1){
                            Imgproc.circle(rgbaInnerWindow, pointn[i], 5, new Scalar(200,200,200,255));
                            Imgproc.line(rgbaInnerWindow, pointp[i], pointn[i], new Scalar(0,200,0,255 ), 1);
                            if (pointn[i].x > rectP2.x){rectP2.x = pointn[i].x;}
                            if (pointn[i].y > rectP2.y){rectP2.y = pointn[i].y;}
                            if (pointn[i].x < rectP1.x){rectP1.x = pointn[i].x;}
                            if (pointn[i].y < rectP1.y){rectP1.y = pointn[i].y;}
                        }
                    }

                }
                count++;
                rgbaInnerWindow.release();
                prevRGB = nextRGB;
                rgbaInnerWindow.release();
                break;

            case MainActivity.VIEW_MODE_HOG_RECORD:
                //fallAccDetected = false;
                //fallHogDetected = false;
                //fallDetected = false;
                writeComplete = true;
                mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
                mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_FASTEST);

                // Fall Detection
                double result1 = 0;
                double result2 = 0;
                double[] cellArrayOri = new double[16];
                double[] cellArrayStr = new double[16];

                // double[] lbpArray;
                int  c1 = 0;
                double sumStr = 0;
                double sumOri = 0;
                double meanOri = 0;
                double meanStr = 0;
                double stdOri = 0;
                double stdStr = 0;
                double thrOri, thrStr = 0;

                Mat grad_x = new Mat();
                Mat grad_y = new Mat();
                Mat magGrad = new Mat();
                Mat angleGrad = new Mat();
                Mat hogStr = new Mat();
                Mat hogOri = new Mat();
                Mat fullStrength = new Mat();
                Mat fullOrientation = new Mat();
                Mat cellStrength = new Mat();
                Mat cellOrientation = new Mat();
                Mat prevStrength = new Mat();
                Mat prevOrientation = new Mat();
                Core.MinMaxLocResult resultArray = new Core.MinMaxLocResult();

                height = rows/4;
                width = cols/4;
                //  double[][] lbpWindow = new double[height][width];
                top = 0;
                left = 0;

                // Generate grad_x and grad_y
                for ( int h=0; h<4; h++){
                    for( int w=0; w<4; w++){
                        cellWindow = rgba.submat(height*h, height*(h+1), width*w, (w+1)*width);
        			  /*for ( int m =0;m<height;m++){
                		  for( int n=0;n<width;n++){
                			  double[] tmp = cellWindow.get(m,n);
                			  lbpWindow[m][n] = tmp[0];
                		  }
                	  }
        			  LBP lbp = new LBP(8,1);
        			  byte[][] resultLBP = lbp.getLBP(cellWindow);
        			  byte[][] resultLBP = lbp.getLBP(lbpArray);*/

                        //Gradient X
                        Imgproc.Sobel( cellWindow, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT );

                        // Gradient Y
                        Imgproc.Sobel( cellWindow, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT );

                        Core.cartToPolar(grad_x, grad_y, magGrad, angleGrad, true);
                        resultArray = Core.minMaxLoc(magGrad);
                        mStrength = new MatOfFloat(0f, (float)resultArray.maxVal+2);//resultArray.maxVal

                        //Turn gradient into histograms
                        Imgproc.calcHist(Arrays.asList(magGrad), mChannels[0], mMat0, hogStr, mHistSizeS, mStrength,false);
                        Imgproc.calcHist(Arrays.asList(angleGrad), mChannels[0], mMat0, hogOri, mHistSizeO, mOrientation, false);
                        fullStrength.push_back(hogStr);
                        fullOrientation.push_back(hogOri);
                        resultArray = Core.minMaxLoc(hogOri);
                        cellArrayOri[c1] = resultArray.maxVal;
                        resultArray = Core.minMaxLoc(hogOri);
                        cellArrayStr[c1] = resultArray.maxVal;
                        c1++;
                    }
                }

                for(int i =0; i < 16;i++){
                    sumOri += cellArrayOri[i];
                    sumStr += cellArrayStr[i];
                }

                meanOri = sumOri/16;
                meanStr = sumStr/16;

                for (int i=0; i < 16;i++) {
                    stdOri += Math.pow(cellArrayOri[i] - meanOri, 2);
                    stdStr += Math.pow(cellArrayStr[i] - meanStr, 2);
                }

                stdOri = Math.sqrt(stdOri/16);
                thrOri =  meanOri-0.5*stdOri;

                stdStr = Math.sqrt(stdStr/16);
                thrStr =  meanStr-0.5*stdStr;

                if (count > 0){
                    for (int i=0; i < 16;i++){
                        if (cellArrayStr[i] > thrStr){
                            cellStrength.push_back(fullStrength.submat(i*18,(i+1)*18,0,1)); 	   //18
                            prevStrength.push_back(strengthNext.submat(i*18, (i+1)*18,0,1)); 	   //18
                        }
                        if (cellArrayOri[i] > thrOri){
                            cellOrientation.push_back(fullOrientation.submat(i*9, (i+1)*9,0,1));
                            prevOrientation.push_back(orientationNext.submat(i*9, (i+1)*9,0,1));
                        }

                    }
                    result1 = 1-Imgproc.compareHist(prevOrientation, cellOrientation, Imgproc.CV_COMP_CORREL);
                    result2 = 1-Imgproc.compareHist(prevStrength, cellStrength, Imgproc.CV_COMP_CORREL);
                    //fallResult = Math.pow(result1*result2, 2);
                    fallResult = result1*result2;
                }

                rgbaInnerWindow = rgba.submat(0, rows, 0, cols);//String str3 = String.valueOf(linear_acceleration[2]);

                // Write different parameters to screen
                /*String str1 = String.valueOf(magMax);
                String str2 = String.valueOf(fallMax);
                String str3 = String.valueOf(finalMax);
                String str4 = String.valueOf(fallAccDetected);
                String str5 = String.valueOf(fallHogDetected);
                String str6 = String.valueOf(fallDetected);
                String text1 = new String("mag: ");
                String text2 = new String("camera: ");

                String text3 = new String("fusion: ");
                String text4 = new String("AccRes: ");
                String text5 = new String("HogRes: ");
                String text6 = new String("Final: ");
                Imgproc.putText(rgbaInnerWindow, text1.concat(str1), new Point(10,10), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,0,0,255));
                Imgproc.putText(rgbaInnerWindow, text2.concat(str2), new Point(10,30), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,200,0,255));
                Imgproc.putText(rgbaInnerWindow, text3.concat(str3), new Point(10,50), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,200,200,255));
                Imgproc.putText(rgbaInnerWindow, text4.concat(str4), new Point(10,80), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,0,50,255));
                Imgproc.putText(rgbaInnerWindow, text5.concat(str5), new Point(10,100), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,0,50,255));
                Imgproc.putText(rgbaInnerWindow, text6.concat(str6), new Point(10,120), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(200,0,50,255));

                if(fallDetected){
                    Imgproc.putText(rgbaInnerWindow,"!!FALL DETECTED!!", new Point(10,220), Core.FONT_HERSHEY_TRIPLEX, 1, new Scalar(200,200,200,255));
                    try {
                        writeCsvData(countFrame, fallAccDetected, fallHogDetected, fallDetected, magnitude, fallResult, magnitude+fallResult);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }*/
                strengthNext = fullStrength;
                orientationNext = fullOrientation;
                // Save image to folder
                SaveImage(rgba, countFrame, saveCount);
                // Save the variables
                try {
                    writeCsvData(countFrame, linear_acceleration[0], linear_acceleration[1], linear_acceleration[2], result1, result2);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                count++;
                countFrame++;
                rgbaInnerWindow.release();
                break;
        }
        return rgba;
    }
}
