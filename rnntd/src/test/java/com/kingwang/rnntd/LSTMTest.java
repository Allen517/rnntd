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

import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.cell.impl.InputNeuron;
import com.kingwang.rnntd.cell.impl.LSTM;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.Activer;
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

	private int inSize;
	private int tmFeatSize;
	private int outSize; 
	private int nodeSize;
	private RNNCell rcell;
	private Map<String, DoubleMatrix> acts = new HashMap<>();
	private InputNeuron input;
	private List<Double> y1_arr = new ArrayList<>();
	private List<Double> y0_arr = new ArrayList<>();
	
	@Before
	public void setCell() {
		// set basic parameters
		inSize = 10;
		tmFeatSize = 1;
		outSize = 8;
		nodeSize = 5;
		
		DoubleMatrix nodeVec = new DoubleMatrix(3, 10);
		List<String> nodeDict = new ArrayList<>();
		nodeDict.add("0");
		nodeDict.add("1");
		nodeDict.add("2");
		
		input = new InputNeuron(nodeVec, nodeDict);
		AlgCons.rnnType="lstm";
		rcell = new LSTM(inSize, tmFeatSize, outSize, nodeSize, new MatIniter(Type.Test, 1., 0, 0)); // set cell
	}
	
	/**
	 * Test method for {@link com.kingwang.rnntd.rnn.impl.LSTM.lstm.Cell#active(int, java.util.Map)}.
	 */
	@Test
	public void testActive() {
		// settings
		LSTM cell = (LSTM)rcell;
		
		System.out.println("active function test");
		input.repMatrix.put(1, 2, 1);
		input.tmFeat = DoubleMatrix.ones(1);
		acts.put("i" + 0, DoubleMatrix.ones(1, 8).mul(0.1));
        acts.put("f" + 0, DoubleMatrix.ones(1, 8).mul(0.2));
        acts.put("gc" + 0, DoubleMatrix.ones(1, 8).mul(0.3));
        acts.put("c" + 0, DoubleMatrix.ones(1, 8).mul(0.4));
        acts.put("o" + 0, DoubleMatrix.ones(1, 8).mul(0.3));
        acts.put("gh" + 0, DoubleMatrix.ones(1, 8).mul(0.2));
        acts.put("h" + 0, DoubleMatrix.ones(1, 8).mul(0.1));
        
		cell.active(1, input, 1, acts);
		System.out.println(Math.pow(1+Math.exp(-1.72), -1)+","+acts.get("i"+1).get(1, 2));
		assertEquals(Math.pow(1+Math.exp(-1.72), -1), acts.get("i"+1).get(1, 2), 10e-3);
		assertEquals(Math.pow(1+Math.exp(-1.72), -1), acts.get("f"+1).get(1, 2), 10e-3);
		assertEquals(Math.tanh(.66), acts.get("gc"+1).get(1, 2), 10e-3);
		double c = .4*Math.pow(1+Math.exp(-1.72), -1)+Math.pow(1+Math.exp(-1.72), -1)*Math.tanh(.66);
		System.out.println(c+","+acts.get("c"+1).get(1, 2));
		assertEquals(c, acts.get("c"+1).get(1, 2), 10e-3);
		DoubleMatrix cMat = DoubleMatrix.ones(1, 8).mul(c);
		double o = Math.pow(1+Math.exp(-(cMat.mmul(cell.Wco).get(2)+.76)), -1);
		System.out.println(o+","+acts.get("o"+1).get(1,2));
		assertEquals(o, acts.get("o"+1).get(1, 2), 10e-3);
		System.out.println(Math.tanh(c)+","+acts.get("gh"+1).get(1, 2));
		assertEquals(Math.tanh(c), acts.get("gh"+1).get(1, 2), 10e-3);
		assertEquals(o*Math.tanh(c), acts.get("h"+1).get(1, 2), 10e-3);
	}

	private void calcOneTurn(List<Integer> nodes) {
		
		for (int t=0; t<nodes.size()-1; t++) {
	    	int ndIdx = nodes.get(t);
	    	int nxtNdIdx = nodes.get(t+1);
	
	        rcell.active(t, input, ndIdx, acts);
	       
	        DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
            acts.put("lambda" + t, MatrixFunctions.exp(d));
            
            DoubleMatrix hatYt = rcell.yDecode(acts.get("h" + t));
	        DoubleMatrix predictYt = Activer.softmax(hatYt);
	        acts.put("py" + t, predictYt);
	        DoubleMatrix y = new DoubleMatrix(1, hatYt.columns);
	        y.put(nxtNdIdx, 1);
	        acts.put("y" + t, y);
		}
	}
	
	private void gradientTestAndretActualGradient(List<Integer> nodes, List<Double> tmList
								, DoubleMatrix mat, int reviseLoc, int targetT, double delta) {
		
		y0_arr = new ArrayList<>();
		y1_arr = new ArrayList<>();
		
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // reset Wxi
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
	    	int nxtNdIdx = nodes.get(t+1);
	    	double tmGap = tmList.get(t);
	    	DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
			DoubleMatrix py = acts.get("py" + t);
			y1_arr.add(logft+Math.log(py.get(nxtNdIdx)));
//			pd.put(nxtNdIdx, pd.get(nxtNdIdx)+Math.log(lambda.get(nxtNdIdx)));
//			y1_arr.add(pd);
		}
		//original
		mat = mat.put(reviseLoc, mat.get(reviseLoc)+2*delta);
		calcOneTurn(nodes);
		for(int t=0; t<targetT+1; t++) {
			int nxtNdIdx = nodes.get(t+1);
			double tmGap = tmList.get(t);
			DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
			DoubleMatrix py = acts.get("py" + t);
			y0_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}
		 
		// test
		mat = mat.put(reviseLoc, mat.get(reviseLoc)-delta); // set back to the original Wxi
		calcOneTurn(nodes);
		rcell.bptt(input, nodes.subList(1, nodes.size()), tmList, acts, nodes.size()-2);
	}
	
	private void gradientTestAndretActualGradientOnX(List<Integer> nodes, List<Double> tmList
							, int targetRow, int reviseLoc, int targetT, double delta) {

		y0_arr = new ArrayList<>();
		y1_arr = new ArrayList<>();

		input.repMatrix.put(targetRow, reviseLoc, input.repMatrix.get(targetRow, reviseLoc)-delta); // reset x
		calcOneTurn(nodes);
		for (int t = 0; t < targetT + 1; t++) {
			int nxtNdIdx = nodes.get(t+1);
			double tmGap = tmList.get(t);
			DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
			DoubleMatrix py = acts.get("py" + t);
			y1_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}
		// original
		input.repMatrix.put(targetRow, reviseLoc, input.repMatrix.get(targetRow, reviseLoc)+2*delta);
		calcOneTurn(nodes);
		for (int t = 0; t < targetT + 1; t++) {
			int nxtNdIdx = nodes.get(t+1);
			double tmGap = tmList.get(t);
			DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
			DoubleMatrix py = acts.get("py" + t);
			y0_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}

		// test
		input.repMatrix.put(targetRow, reviseLoc, input.repMatrix.get(targetRow, reviseLoc) - delta); // set back to the original x
		calcOneTurn(nodes);
		input.bptt(rcell, nodes, tmList, acts, targetT);
		
	}
	
	private void gradientTestAndretActualGradientOnw(List<Integer> nodes,
			List<Double> tmList, DoubleMatrix mat, int reviseLoc, int targetT,
			double delta) {

		y0_arr = new ArrayList<>();
		y1_arr = new ArrayList<>();

		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc) - delta); // reset Wxi
		calcOneTurn(nodes);
		for (int t = 0; t < targetT + 1; t++) {
			int nxtNdIdx = nodes.get(t + 1);
			double tmGap = tmList.get(t);
			DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
			DoubleMatrix py = acts.get("py" + t);
			y1_arr.add(logft+Math.log(py.get(nxtNdIdx)));
		}
		// original
		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc) + 2 * delta);
		calcOneTurn(nodes);
		for (int t = 0; t < targetT + 1; t++) {
			int nxtNdIdx = nodes.get(t + 1);
			double tmGap = tmList.get(t);
			DoubleMatrix d = rcell.dDecode(acts.get("h" + t));
			DoubleMatrix lambda = acts.get("lambda" + t);
			double logft = .0;
			if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
            			+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap))
            			-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            if(AlgCons.tmDist.equalsIgnoreCase("const")) {
                logft = d.get(nxtNdIdx)-lambda.sum()*tmGap
                		-AlgCons.gamma*MatrixFunctions.abs(lambda).sum();
            }
            DoubleMatrix py = acts.get("py" + t);
			y0_arr.add(logft + Math.log(py.get(nxtNdIdx)));
		}

		// test
		acts.clear();
		mat = mat.put(reviseLoc, mat.get(reviseLoc) - delta);
		calcOneTurn(nodes);
		rcell.bptt(input, nodes.subList(1, nodes.size()), tmList, acts, targetT);
		input.bptt(rcell, nodes, tmList, acts, targetT);
	}
	
	/**
	 * Test method for {@link com.kingwang.rnntd.rnn.impl.LSTM.lstm.Cell#bptt(java.util.List, java.util.Map, java.util.Map, int, double)}.
	 */
	@Test
	public void testBptt() {
		
		AlgCons.tmDist = "exp";
		double delta = 10e-7;
		
		// set input
		List<Integer> ndList = new ArrayList<>();
		ndList.add(2);
		ndList.add(1);
		ndList.add(0);
		
		List<Double> tmList = new ArrayList<>();
		tmList.add(3.);
		tmList.add(5.);
		
		// set nodeVec
		input.repMatrix.put(2, 2, 1);
		input.repMatrix.put(1, 2, 1);
		input.repMatrix.put(0, 2, 1);
		
		input.tmFeat = DoubleMatrix.ones(1).mul(10);
//		DoubleMatrix x = new DoubleMatrix(1, 10);
//		x.put(2, 1);
//		nodeVec.put(3, x);
//		nodeVec.put(2, x);
//		nodeVec.put(1, x);
		
		rcell = new LSTM(inSize, tmFeatSize, outSize, nodeSize, new MatIniter(Type.Uniform, 0.1, 0, 0));
		int reviseLoc = 2;
		int targetT = 1;
		
		LSTM cell = (LSTM)rcell;
		
		/**
		 * Wxi
		 */
		System.out.println("Wxi test");
		gradientTestAndretActualGradient(ndList, tmList, cell.Wxi, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wxf, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wxo, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Whc, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wxc, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wdi, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wdf, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wdo, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wdc, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Wdc, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Whd, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.bd, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.Why, reviseLoc, targetT, delta);
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
		gradientTestAndretActualGradient(ndList, tmList, cell.by, reviseLoc, targetT, delta);
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
			gradientTestAndretActualGradientOnw(ndList, tmList, input.w, 0, targetT, delta);
			double deltaw = 0;
			for (int t = 0; t < targetT + 1; t++) {
				double tmp = y0_arr.get(t) - y1_arr.get(t);
				deltaw += tmp / 2 / delta;
			}
			System.out.println("deltaw: " + deltaw + "," + (-acts.get("dw").get(0)));
			assertEquals(deltaw, -acts.get("dw").get(0), 10e-7);
		}
		
		/**
		 * x
		 */
		System.out.println("X test");
		int testT = 0;
		
		int testNdIdx = ndList.get(testT);
		gradientTestAndretActualGradientOnX(ndList, tmList, testNdIdx, reviseLoc, targetT, delta);
		double deltaX0_2 = 0; 
		for(int t=testT; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaX0_2 += tmp/2/delta;
		}
		System.out.println("deltaX0_2: "+deltaX0_2+","+(-acts.get("dxMat").get(testNdIdx, reviseLoc)));
		assertEquals(deltaX0_2, -acts.get("dxMat").get(testNdIdx, reviseLoc), 10e-7);
		
		testT = 1;
		testNdIdx = ndList.get(testT);
		gradientTestAndretActualGradientOnX(ndList, tmList, testNdIdx, reviseLoc, targetT, delta);
		double deltaX1_2 = 0; 
		for(int t=testT; t<targetT+1; t++) {
			double tmp = y0_arr.get(t)-y1_arr.get(t);
			deltaX1_2 += tmp/2/delta;
		}
		System.out.println("deltaX1_2: "+deltaX1_2+","+(-acts.get("dxMat").get(testNdIdx, reviseLoc)));
		assertEquals(deltaX1_2, -acts.get("dxMat").get(testNdIdx, reviseLoc), 10e-7);
		
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
