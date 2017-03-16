/**   
 * @package	com.kingwang.rnncdm
 * @File		CellTest.java
 * @Crtdate	May 23, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.junit.Before;
import org.junit.Test;

import com.kingwang.rnntd.cell.impl.GRU;
import com.kingwang.rnntd.cell.impl.InputLayer;
import com.kingwang.rnntd.cell.impl.OutputLayer;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * May 23, 2016 10:16:59 AM
 * @version 1.0
 */
public class GRUTest {

	private int inDynSize;
	private int inFixedSize;
	private int outSize; 
	private int nodeSize;
	private InputLayer input;
	private GRU gru;
	private OutputLayer output;
	private Map<String, DoubleMatrix> nodeCode;
	private Map<String, DoubleMatrix> acts = new HashMap<>();
	private List<Double> y1_arr = new ArrayList<>();
	private List<Double> y0_arr = new ArrayList<>();
	
	@Before
	public void setCell() {
		AlgCons.tmDist = "exp";
		
		// set basic parameters
		inDynSize = 10;
		inFixedSize = 1;
		outSize = 8;
		nodeSize = 5;
		AlgCons.biasInitVal = 0;
		
		AlgCons.rnnType="gru";
		input = new InputLayer(nodeSize, inDynSize, new MatIniter(Type.Test, 0, 0, 0));
		gru = new GRU(inDynSize, inFixedSize, outSize, new MatIniter(Type.Test, 0, 0, 0)); // set cell
		output = new OutputLayer(outSize, nodeSize, new MatIniter(Type.Test, 0, 0, 0));
		
		DoubleMatrix Wx = new DoubleMatrix(nodeSize, inDynSize);
		nodeCode = new HashMap<>();
		DoubleMatrix code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("0", code.put(0, 1));
		code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("1", code.put(1, 1));
		code = new DoubleMatrix(1, nodeSize);
		nodeCode.put("2", code.put(2, 1));
	}
	
	private void calcOneTurn(DoubleMatrix nodes) {
		
		for (int t=0; t<nodes.length-1; t++) {
			int ndId = (int) nodes.get(t);
			int nxtNdId = (int) nodes.get(t+1);
	
	    	DoubleMatrix fixedFeat = DoubleMatrix.zeros(1, inFixedSize);
			fixedFeat.put(0, 1);
	    	
			DoubleMatrix code = new DoubleMatrix(1);
			code.put(0, (double)ndId);
	    	acts.put("code"+t, code);
	    	acts.put("fixedFeat"+t, fixedFeat);
	    	
	    	input.active(t, acts);
            gru.active(t, acts);
            output.active(t, acts);
            
            DoubleMatrix y = new DoubleMatrix(1, nodeSize);
            y.put(nxtNdId, 1);
	        acts.put("y" + t, y);
		}
	}
	
	private void gradientTestAndretActualGradient(DoubleMatrix mat, int reviseLoc, int targetT
										, double delta) {
		
		DoubleMatrix tmList = acts.get("tmList");
		DoubleMatrix nodes = acts.get("ndList");
		
		y0_arr = new ArrayList<>();
		y1_arr = new ArrayList<>();
		
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // reset Wxi
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
			int nxtNdIdx = (int) nodes.get(t+1);
			double tmGap = tmList.get(t);
			
	        DoubleMatrix py = acts.get("py"+t);
	        
	        double logft = .0;
	        DoubleMatrix lambda = acts.get("lambda"+t);
	        if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
	        	logft = Math.log(lambda.get(nxtNdIdx))+output.w.get(0)*tmGap
	        			+lambda.sum()/output.w.get(0)*(1-Math.exp(output.w.get(0)*tmGap))
	        			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
	        }
	        if(AlgCons.tmDist.equalsIgnoreCase("const")) {
	            logft = Math.log(lambda.get(nxtNdIdx))-lambda.sum()*tmGap
	            		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
	        }
			y1_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}
		//original
		mat = mat.put(reviseLoc, mat.get(reviseLoc)+2*delta);
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
			int nxtNdIdx = (int) nodes.get(t+1);
			double tmGap = tmList.get(t);
			
	        DoubleMatrix py = acts.get("py"+t);
	        
	        double logft = .0;
	        DoubleMatrix lambda = acts.get("lambda"+t);
	        if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
	        	logft = Math.log(lambda.get(nxtNdIdx))+output.w.get(0)*tmGap
	        			+lambda.sum()/output.w.get(0)*(1-Math.exp(output.w.get(0)*tmGap))
	        			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
	        }
	        if(AlgCons.tmDist.equalsIgnoreCase("const")) {
	            logft = Math.log(lambda.get(nxtNdIdx))-lambda.sum()*tmGap
	            		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
	        }
			y0_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}
		 
		// test
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // set back to the original Wxi
		calcOneTurn(nodes);
		output.bptt(acts, targetT);
		gru.bptt(acts, targetT, output);
		input.bptt(acts, targetT, gru);
	}
	
	/**
	 * Test method for {@link com.kingwang.rnntd.rnn.impl.LSTM.lstm.Cell#bptt(java.util.List, java.util.Map, java.util.Map, int, double)}.
	 */
	@Test
	public void testBptt() {
		
		inDynSize = 3;
		inFixedSize = 1;
		outSize = 2;
		nodeSize = 3;
		
		double delta = 10e-7;
		
		// set input
		DoubleMatrix ndList = new DoubleMatrix(4);
		ndList.add(2);
		ndList.add(1);
		ndList.add(0);
		ndList.add(1);
		acts.put("ndList", ndList);
		
		// set tmList
		DoubleMatrix tmList = new DoubleMatrix(3);
		tmList.put(0, 2.);
		tmList.put(1, 3.);
		tmList.put(2, 4.);
		acts.put("tmList", tmList);
		
		MatIniter initer = new MatIniter(Type.Uniform, 1, 0, 0);
		
		input = new InputLayer(nodeSize, inDynSize, initer);
		gru = new GRU(inDynSize, inFixedSize, outSize, initer);
		output = new OutputLayer(outSize, nodeSize, initer);
		int reviseLoc = 1;
		int targetT = 2;
		
		/**
		 * Wxr
		 */
		System.out.println("Wxr test");
		gradientTestAndretActualGradient(gru.Wxr, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxr_2 += tmp/2/delta;
		}
		System.out.println(deltaWxr_2+","+(-acts.get("dWxr").get(reviseLoc)));
		assertEquals(deltaWxr_2, -acts.get("dWxr").get(reviseLoc), 10e-7);
		
		/***
		 * Whh
		 */
		System.out.println("Whh test");
		gradientTestAndretActualGradient(gru.Whh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhh_2 += tmp/2/delta;
		}
		System.out.println(deltaWhh_2+","+(-acts.get("dWhh").get(reviseLoc)));
		assertEquals(deltaWhh_2, -acts.get("dWhh").get(reviseLoc), 10e-7);
		
		/***
		 * Wxz
		 */
		System.out.println("Wxz test");
		gradientTestAndretActualGradient(gru.Wxz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxz_2 += tmp/2/delta;
		}
		System.out.println(deltaWxz_2+","+(-acts.get("dWxz").get(reviseLoc)));
		assertEquals(deltaWxz_2, -acts.get("dWxz").get(reviseLoc), 10e-7);

		/**
		 * Wxh
		 */
		System.out.println("Wxh test");
		gradientTestAndretActualGradient(gru.Wxh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxh_2 += tmp/2/delta;
		}
		System.out.println(deltaWxh_2+","+(-acts.get("dWxh").get(reviseLoc)));
		assertEquals(deltaWxh_2, -acts.get("dWxh").get(reviseLoc), 10e-7);
		
		/**
		 * Wdr
		 */
		reviseLoc = 0;
		System.out.println("Wdr test");
		gradientTestAndretActualGradient(gru.Wdr, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdr_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdr_2 += tmp/2/delta;
		}
		System.out.println(deltaWdr_2+","+(-acts.get("dWdr").get(reviseLoc)));
		assertEquals(deltaWdr_2, -acts.get("dWdr").get(reviseLoc), 10e-7);
		
		/***
		 * Wdz
		 */
		System.out.println("Wdz test");
		gradientTestAndretActualGradient(gru.Wdz, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdz_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdz_2 += tmp/2/delta;
		}
		System.out.println(deltaWdz_2+","+(-acts.get("dWdz").get(reviseLoc)));
		assertEquals(deltaWdz_2, -acts.get("dWdz").get(reviseLoc), 10e-7);

		/**
		 * Wdh
		 */
		System.out.println("Wdh test");
		gradientTestAndretActualGradient(gru.Wdh, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWdh_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWdh_2 += tmp/2/delta;
		}
		System.out.println(deltaWdh_2+","+(-acts.get("dWdh").get(reviseLoc)));
		assertEquals(deltaWdh_2, -acts.get("dWdh").get(reviseLoc), 10e-7);
		
		/**
		 * Whd
		 */
		System.out.println("Whd test");
		gradientTestAndretActualGradient(output.Whd, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhd_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhd_2 += tmp/2/delta;
		}
		System.out.println(deltaWhd_2+","+(-acts.get("dWhd").get(reviseLoc)));
		assertEquals(deltaWhd_2, -acts.get("dWhd").get(reviseLoc), 10e-7);
		
		/**
		 * bd
		 */
		System.out.println("bd test");
		gradientTestAndretActualGradient(output.bd, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabd_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabd_2 += tmp/2/delta;
		}
		System.out.println(deltabd_2+","+(-acts.get("dbd").get(0)));
		assertEquals(deltabd_2, -acts.get("dbd").get(0), 10e-7);
		
		/**
		 * Why
		 */
		System.out.println("Why test");
		gradientTestAndretActualGradient(output.Why, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhy_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhy_2 += tmp/2/delta;
		}
		System.out.println(deltaWhy_2+","+(-acts.get("dWhy").get(reviseLoc)));
		assertEquals(deltaWhy_2, -acts.get("dWhy").get(reviseLoc), 10e-7);
		
		/**
		 * by
		 */
		System.out.println("by test");
		gradientTestAndretActualGradient(output.by, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaby_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaby_2 += tmp/2/delta;
		}
		System.out.println(deltaby_2+","+(-acts.get("dby").get(0)));
		assertEquals(deltaby_2, -acts.get("dby").get(0), 10e-7);
		
		/**
		 * w
		 */
		if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
			System.out.println("w test");
			gradientTestAndretActualGradient(output.w, 0, targetT, delta);
			double deltaw = 0;
			for (int t = 0; t < targetT + 1; t++) {
				double tmp = y0_arr.get(t) - y1_arr.get(t);
				deltaw += tmp / 2 / delta;
			}
			System.out.println("deltaw: " + deltaw + "," + (-acts.get("dw").get(0)));
			assertEquals(deltaw, -acts.get("dw").get(0), 10e-7);
		}
		
		/**
		 * Wx
		 */
		System.out.println("Wx test");
		reviseLoc = 2;
		gradientTestAndretActualGradient(input.Wx, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWx_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWx_2 += tmp/2/delta;
		}
		System.out.println(deltaWx_2+","+(-acts.get("dWx").get(reviseLoc)));
		assertEquals(deltaWx_2, -acts.get("dWx").get(reviseLoc), 10e-7);
		
		/**
		 * bx
		 */
		System.out.println("bx test");
		gradientTestAndretActualGradient(input.bx, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltabx_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltabx_2 += tmp/2/delta;
		}
		System.out.println(deltabx_2+","+(-acts.get("dbx").get(reviseLoc)));
		assertEquals(deltabx_2, -acts.get("dbx").get(reviseLoc), 10e-7);
		
	}
	
//
//	/**
//	 * Test method for {@link com.kingwang.rnncdm.lstm.Cell#decode(org.jblas.DoubleMatrix)}.
//	 */
//	@Test
//	public void testDecode() {
//		fail("Not yet implemented");
//	}
//
//	/**
//	 * Test method for {@link com.kingwang.rnncdm.lstm.Cell#loadRNNModel(java.lang.String)}.
//	 */
//	@Test
//	public void testLoadRNNModel() {
//		fail("Not yet implemented");
//	}

}
