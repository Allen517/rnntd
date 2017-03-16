/**   
 * @package	com.kingwang.cdmrnn.utils
 * @File		AttBatchDerivative.java
 * @Crtdate	Oct 2, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.batchderv.impl;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.batchderv.BatchDerivative;
import com.kingwang.rnntd.cons.AlgCons;

/**
 *
 * @author King Wang
 * 
 * Oct 2, 2016 8:29:49 PM
 * @version 1.0
 */
public class OutputBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6945194770004264043L;

	public DoubleMatrix dWhy;
	public DoubleMatrix dby;
	
	public DoubleMatrix dWhd;
	public DoubleMatrix dbd;
	
	public DoubleMatrix dw;
	
	public void clearBatchDerv() {
		dWhy = null;
		dby = null;
		
		dWhd = null;
		dbd = null;
		
		dw = null;
	}

	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		DoubleMatrix _dWhy = acts.get("dWhy");
		DoubleMatrix _dby = acts.get("dby");
		
		DoubleMatrix _dWhd = acts.get("dWhd");
		DoubleMatrix _dbd = acts.get("dbd");
		
		if(dWhy==null) {
			dWhy = new DoubleMatrix(_dWhy.rows, _dWhy.columns);
		}
		if(dby==null) {
			dby = new DoubleMatrix(_dby.rows, _dby.columns);
		}
		if(dWhd==null) {
			dWhd = new DoubleMatrix(_dWhd.rows, _dWhd.columns);
		}
		if(dbd==null) {
			dbd = new DoubleMatrix(_dbd.rows, _dbd.columns);
		}
		
		dWhy = dWhy.add(_dWhy).mul(avgFac);
		dby = dby.add(_dby).mul(avgFac);
		
		dWhd = dWhd.add(_dWhd).mul(avgFac);
		dbd = dbd.add(_dbd).mul(avgFac);
		
		if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
			DoubleMatrix _dw = acts.get("dw");
			if(dw==null) {
				dw = new DoubleMatrix(_dw.rows, _dw.columns);
			}
			dw = dw.add(_dw).mul(avgFac);
		}
	}

}
