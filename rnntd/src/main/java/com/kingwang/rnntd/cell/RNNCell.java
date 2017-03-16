/**   
 * @package	com.kingwang.ctsrnn.lstm
 * @File		RNNCell.java
 * @Crtdate	Jul 3, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.cell;

import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.batchderv.BatchDerivative;
import com.kingwang.rnntd.cell.impl.InputNeuron;

/**
 *
 * @author King Wang
 * 
 * Jul 3, 2016 10:25:29 PM
 * @version 1.0
 */
public interface RNNCell {
	
	public DoubleMatrix yDecode(DoubleMatrix ht);
	
	public DoubleMatrix dDecode(DoubleMatrix ht);
	
	/**
	 * activation function
	 * 
	 * @param t
	 * @param input
	 * @param node must be the index of current node
	 * @param acts
	 */
	public void active(int t, InputNeuron input, int node, Map<String, DoubleMatrix> acts);

	/**
	 * back-propagation through time
	 * 
	 * @param input
	 * @param ndList must be the list of predicting nodes
	 * @param tmList must be the list of time gap between current node and correpsonding predicting node
	 * @param acts
	 * @param lastT
	 */
	public void bptt(InputNeuron input, List<Integer> ndList, List<Double> tmList
			, Map<String, DoubleMatrix> acts, int lastT);
	
	public void updateParametersByAdaGrad(BatchDerivative batchDerv, double lr);
	
	public void updateParametersByAdam(BatchDerivative batchDerv, double lr
			, double beta1, double beta2, int epochT);
	
	public void writeRes(String outFile);
	
	public void loadRNNModel(String rnnModelFile);
}
