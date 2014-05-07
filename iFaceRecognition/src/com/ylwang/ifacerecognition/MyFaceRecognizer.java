package com.ylwang.ifacerecognition;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.opencv.core.Mat;

import android.os.Environment;
import android.util.Log;

public class MyFaceRecognizer {
	private String TAG = "MyFaceRecognizer";

	public MyFaceRecognizer() {
		mNativeObj = nativeCreativeModel("FaceRecognizer::Eigenfaces");
		mFaceRec = new FaceRecExt(mNativeObj);
		mModelName = "ylwang_FaceRec.xml";
		File FaceModel = new File(Environment.getExternalStorageDirectory(),mModelName);
		
		try {
			if (FaceModel.exists())
				Log.i(TAG,mModelName+"Exits");
			else
				FaceModel.createNewFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		mModelPath=FaceModel.getAbsolutePath();
		Log.i(TAG, "The MyFaceRecognizer Created.");

	}

	public void load(String filename) {

		mFaceRec.load(filename);

		return;
	}

	public void save(String filename) {

		mFaceRec.save(filename);

		return;
	}

	public void train(List<Mat> src, Mat labels) {

		mFaceRec.train(src, labels);
		save(mModelPath);
		Log.i(TAG, "SuccessFully Trained");
	}

	public void update(List<Mat> src, Mat labels) {
		
		mFaceRec.update(src, labels);
		save(mModelPath);
		Log.i(TAG, "SuccessFully Updated");
	}

	public void predict(Mat src, int[] label, double[] confidence) {
		//load(mModelPath);
		mFaceRec.predict(src, label, confidence);
		Log.i(TAG, "SuccessFully Predicted");

	}
	public double getSimilarity(Mat imgA,Mat imgB) {
		Log.i(TAG, "getSimilarity");
		return nativeGetSimilarity(imgA.getNativeObjAddr(),imgB.getNativeObjAddr());
		

	}

	public void test() {
		Log.i(TAG, "The MyFaceRecognizer Works Normally.");
	}

	private long mNativeObj = 0;
	private FaceRecExt mFaceRec=null;
	private String mModelName;
	private String mModelPath;
	private native long nativeCreativeModel(String Name);
	private native double nativeGetSimilarity(long ImgA,long ImgB);
}
