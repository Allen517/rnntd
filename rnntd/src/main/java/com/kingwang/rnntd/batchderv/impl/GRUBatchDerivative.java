/**   
 * @package	com.kingwang.rnncdm.layers
 * @File		InputNeuron.java
 * @Crtdate	May 17, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.batchderv.impl;

import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.batchderv.BatchDerivative;

/**
 *
 * @author King Wang
 * 
 * May 17, 2016 8:49:16 PM
 * @version 1.0
 */
public class GRUBatchDerivative implements BatchDerivative, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3105102557880200595L;
	
	public DoubleMatrix dWxr;
	public DoubleMatrix dWdr;
	public DoubleMatrix dWhr;
	public DoubleMatrix dbr;
    
	public DoubleMatrix dWxz;
	public DoubleMatrix dWdz;
	public DoubleMatrix dWhz;
	public DoubleMatrix dbz;
    
	public DoubleMatrix dWxh;
	public DoubleMatrix dWdh;
	public DoubleMatrix dWhh;
	public DoubleMatrix dbh;
    
	public DoubleMatrix dWhd;
	public DoubleMatrix dbd;
	
	public DoubleMatrix dWhy;
	public DoubleMatrix dby;
	
	public DoubleMatrix dxMat;
	
	public DoubleMatrix dw;
	
	public void clearBatchDerv() {
		
		dWxr = null;
		dWdr = null;
		dWhr = null;
		dbr = null;
		
		dWxz = null;
		dWdz = null;
		dWhz = null;
		dbz = null;
		
		dWxh = null;
		dWdh = null;
		dWhh = null;
		dbh = null;
		
		dWhd = null;
		dbd = null;
		
		dWhy = null;
		dby = null;
		
		dxMat = null;
		
		dw = null;
	}
	
	public void batchDervCalc(Map<String, DoubleMatrix> acts, double avgFac) {
		
		DoubleMatrix _dWxr = acts.get("dWxr");
		DoubleMatrix _dWdr = acts.get("dWdr");
		DoubleMatrix _dWhr = acts.get("dWhr");
		DoubleMatrix _dbr = acts.get("dbr");
		
		DoubleMatrix _dWxz = acts.get("dWxz");
		DoubleMatrix _dWdz = acts.get("dWdz");
		DoubleMatrix _dWhz = acts.get("dWhz");
		DoubleMatrix _dbz = acts.get("dbz");
		
		DoubleMatrix _dWxh = acts.get("dWxh");
		DoubleMatrix _dWdh = acts.get("dWdh");
		DoubleMatrix _dWhh = acts.get("dWhh");
		DoubleMatrix _dbh = acts.get("dbh");
		
		DoubleMatrix _dWhd = acts.get("dWhd");
		DoubleMatrix _dbd = acts.get("dbd");
		
		DoubleMatrix _dWhy = acts.get("dWhy");
		DoubleMatrix _dby = acts.get("dby");
		
		DoubleMatrix _dxMat = acts.get("dxMat");
		
		DoubleMatrix _dw = acts.get("dw");
		
		if(dWxr==null) {
			dWxr = new DoubleMatrix(_dWxr.rows, _dWxr.columns);
		}
		if(dWdr==null) {
			dWdr = new DoubleMatrix(_dWdr.rows, _dWdr.columns);
		}
		if(dWhr==null) {
			dWhr = new DoubleMatrix(_dWhr.rows, _dWhr.columns);
		}
		if(dbr==null) {
			dbr = new DoubleMatrix(_dbr.rows, _dbr.columns);
		}
		
		if(dWxz==null) {
			dWxz = new DoubleMatrix(_dWxz.rows, _dWxz.columns);
		}
		if(dWdz==null) {
			dWdz = new DoubleMatrix(_dWdz.rows, _dWdz.columns);
		}
		if(dWhz==null) {
			dWhz = new DoubleMatrix(_dWhz.rows, _dWhz.columns);
		}
		if(dbz==null) {
			dbz = new DoubleMatrix(_dbz.rows, _dbz.columns);
		}
		
		if(dWxh==null) {
			dWxh = new DoubleMatrix(_dWxh.rows, _dWxh.columns);
		}
		if(dWdh==null) {
			dWdh = new DoubleMatrix(_dWdh.rows, _dWdh.columns);
		}
		if(dWhh==null) {
			dWhh = new DoubleMatrix(_dWhh.rows, _dWhh.columns);
		}
		if(dbh==null) {
			dbh = new DoubleMatrix(_dbh.rows, _dbh.columns);
		}
		
		if(dWhd==null) {
			dWhd = new DoubleMatrix(_dWhd.rows, _dWhd.columns);
		}
		if(dbd==null) {
			dbd = new DoubleMatrix(_dbd.rows, _dbd.columns);
		}
		
		if(dWhy==null) {
			dWhy = new DoubleMatrix(_dWhy.rows, _dWhy.columns);
		}
		if(dby==null) {
			dby = new DoubleMatrix(_dby.rows, _dby.columns);
		}
		
		if(dxMat==null) {
			dxMat = new DoubleMatrix(_dxMat.rows, _dxMat.columns);
		}
		
		if(dw==null) {
			dw = new DoubleMatrix(_dw.rows, _dw.columns);
		}
		
		dWxr = dWxr.add(_dWxr).mul(avgFac);
		dWdr = dWdr.add(_dWdr).mul(avgFac);
		dWhr = dWhr.add(_dWhr).mul(avgFac);
		dbr = dbr.add(_dbr).mul(avgFac);
		
		dWxz = dWxz.add(_dWxz).mul(avgFac);
		dWdz = dWdz.add(_dWdz).mul(avgFac);
		dWhz = dWhz.add(_dWhz).mul(avgFac);
		dbz = dbz.add(_dbz).mul(avgFac);
		
		dWxh = dWxh.add(_dWxh).mul(avgFac);
		dWdh = dWdh.add(_dWdh).mul(avgFac);
		dWhh = dWhh.add(_dWhh).mul(avgFac);
		dbh = dbh.add(_dbh).mul(avgFac);
		
		dWhd = dWhd.add(_dWhd).mul(avgFac);
		dbd = dbd.add(_dbd).mul(avgFac);
		
		dWhy = dWhy.add(_dWhy).mul(avgFac);
		dby = dby.add(_dby).mul(avgFac);
		
		dxMat = dxMat.add(_dxMat).mul(avgFac);
		
		dw = dw.add(_dw).mul(avgFac);
	}
	
	public DoubleMatrix getdxMat() {
		return dxMat;
	}

	/* (non-Javadoc)
	 * @see com.kingwang.rnntd.utils.BatchDerivative#getdw()
	 */
	@Override
	public DoubleMatrix getdw() {
		return dw;
	}
	
}
