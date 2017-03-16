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

import com.kingwang.rnntd.cell.impl.InputLayer;
import com.kingwang.rnntd.cell.impl.LSTM;
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
public class LSTMTest {

	private int inDynSize;
	private int inFixedSize;
	private int outSize; 
	private int nodeSize;
	private InputLayer input;
	private LSTM lstm;
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
		
		AlgCons.rnnType="lstm";
		input = new InputLayer(nodeSize, inDynSize, new MatIniter(Type.Test, 0, 0, 0));
		lstm = new LSTM(inDynSize, inFixedSize, outSize, new MatIniter(Type.Test, 0, 0, 0)); // set cell
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
            lstm.active(t, acts);
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
		lstm.bptt(acts, targetT, output);
		input.bptt(acts, targetT, lstm);
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
		lstm = new LSTM(inDynSize, inFixedSize, outSize, initer);
		output = new OutputLayer(outSize, nodeSize, initer);
		int reviseLoc = 1;
		int targetT = 2;
		
		/**
		 * Wxi
		 */
		System.out.println("Wxi test");
		gradientTestAndretActualGradient(lstm.Wxi, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxi_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxi_2 += tmp/2/delta;
		}
		System.out.println(deltaWxi_2+","+(-acts.get("dWxi").get(reviseLoc)));
		assertEquals(deltaWxi_2, -acts.get("dWxi").get(reviseLoc), 10e-7);
		
		/***
		 * Wxf
		 */
		System.out.println("Wxf test");
		gradientTestAndretActualGradient(lstm.Wxf, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxf_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxf_2 += tmp/2/delta;
		}
		System.out.println(deltaWxf_2+","+(-acts.get("dWxf").get(reviseLoc)));
		assertEquals(deltaWxf_2, -acts.get("dWxf").get(reviseLoc), 10e-7);

		/**
		 * Wxo
		 */
		System.out.println("Wxo test");
		gradientTestAndretActualGradient(lstm.Wxo, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxo_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxo_2 += tmp/2/delta;
		}
		System.out.println(deltaWxo_2+","+(-acts.get("dWxo").get(reviseLoc)));
		assertEquals(deltaWxo_2, -acts.get("dWxo").get(reviseLoc), 10e-7);
		
		/**
		 * Whc
		 */
		System.out.println("Whc test");
		gradientTestAndretActualGradient(lstm.Whc, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWhc_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhc_2 += tmp/2/delta;
		}
		System.out.println(deltaWhc_2+","+(-acts.get("dWhc").get(reviseLoc)));
		assertEquals(deltaWhc_2, -acts.get("dWhc").get(reviseLoc), 10e-7);
		
		/**
		 * Wxc
		 */
		System.out.println("Wxc test");
		gradientTestAndretActualGradient(lstm.Wxc, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		double deltaWxc_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxc_2 += tmp/2/delta;
		}
		System.out.println(deltaWxc_2+","+(-acts.get("dWxc").get(reviseLoc)));
		assertEquals(deltaWxc_2, -acts.get("dWxc").get(reviseLoc), 10e-7);
		
		/**
		 * Wdi
		 */
		System.out.println("Wdi test");
		gradientTestAndretActualGradient(lstm.Wdi, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		deltaWxi_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxi_2 += tmp/2/delta;
		}
		System.out.println(deltaWxi_2+","+(-acts.get("dWdi").get(reviseLoc)));
		assertEquals(deltaWxi_2, -acts.get("dWdi").get(reviseLoc), 10e-7);
		
		/***
		 * Wdf
		 */
		System.out.println("Wdf test");
		gradientTestAndretActualGradient(lstm.Wdf, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		deltaWxf_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxf_2 += tmp/2/delta;
		}
		System.out.println(deltaWxf_2+","+(-acts.get("dWdf").get(reviseLoc)));
		assertEquals(deltaWxf_2, -acts.get("dWdf").get(reviseLoc), 10e-7);

		/**
		 * Wdo
		 */
		System.out.println("Wdo test");
		gradientTestAndretActualGradient(lstm.Wdo, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		deltaWxo_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxo_2 += tmp/2/delta;
		}
		System.out.println(deltaWxo_2+","+(-acts.get("dWdo").get(reviseLoc)));
		assertEquals(deltaWxo_2, -acts.get("dWdo").get(reviseLoc), 10e-7);
		
		/**
		 * Wdc
		 */
		System.out.println("Wdc test");
		gradientTestAndretActualGradient(lstm.Wdc, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		deltaWhc_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWhc_2 += tmp/2/delta;
		}
		System.out.println(deltaWhc_2+","+(-acts.get("dWdc").get(reviseLoc)));
		assertEquals(deltaWhc_2, -acts.get("dWdc").get(reviseLoc), 10e-7);
		
		/**
		 * Wdc
		 */
		System.out.println("Wdc test");
		gradientTestAndretActualGradient(lstm.Wdc, reviseLoc, targetT, delta);
		// get the actual partial y/partial x
		deltaWxc_2 = 0;
		for(int t=0; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaWxc_2 += tmp/2/delta;
		}
		System.out.println(deltaWxc_2+","+(-acts.get("dWdc").get(reviseLoc)));
		assertEquals(deltaWxc_2, -acts.get("dWdc").get(reviseLoc), 10e-7);
		
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
		System.out.println(deltabd_2+","+(-acts.get("dbd").get(reviseLoc)));
		assertEquals(deltabd_2, -acts.get("dbd").get(reviseLoc), 10e-7);
		
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
		System.out.println(deltaby_2+","+(-acts.get("dby").get(reviseLoc)));
		assertEquals(deltaby_2, -acts.get("dby").get(reviseLoc), 10e-7);
		
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
