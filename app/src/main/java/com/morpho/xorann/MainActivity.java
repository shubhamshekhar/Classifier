package com.morpho.xorann;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private TensorFlowInferenceInterface inferenceInterface;
    long time;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load model from assets
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "model.h5.pb");
        time = System.currentTimeMillis();
        float [] input = {48,30,40,1};//kjjjgit a
        float [] output = predict(input);
        Log.d("Score3", Arrays.toString(input)+" -> "+Arrays.toString(output));


    }

    private float[] predict(float[] input){
        // model has only 1 output neuron
        float output[] = new float[4];

        // feed network with input of shape (1,input.length) = (1,2)
        inferenceInterface.feed("dense_10_input_1", input, 1, input.length);
        inferenceInterface.run(new String[]{"output_node0"});
        inferenceInterface.fetch("output_node0", output);
        time = System.currentTimeMillis() - time;
        Log.v("Score3",String.valueOf(time));

        // return prediction
        return output;
    }
}
