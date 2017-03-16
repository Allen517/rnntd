/**   
 * @package	com.kingwang.rnncdm.evals
 * @File		RNNModelMRREvals.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.evals;

import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.cell.impl.InputNeuron;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.dataset.SeqLoader;
import com.kingwang.rnntd.utils.Activer;
import com.kingwang.rnntd.utils.LossFunction;
import com.kingwang.rnntd.utils.TmFeatExtractor;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:03:33 PM
 * @version 1.0
 */
public class RNNModelEvals {
	
	public static double validationOnIntegration(InputNeuron input, RNNCell cell, int nodeSize
			, List<String> crsValSeq, OutputStreamWriter oswLog) {

		double logLkHd = .0;
		double mrr = .0;
		for (String seq : crsValSeq) {
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq,
					input.nodeDict);
			double cas_logLkHd = 0;
			double cas_mrr = 0;
			double prevTm = 0;
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				String curNd = curInfo[0];
				int curNdIdx = input.nodeDict.indexOf(curNd);
				String nxtNd = nextInfo[0];
				int nxtNdIdx = input.nodeDict.indexOf(nxtNd);
				if (nxtNdIdx < 0 || curNdIdx < 0) {
					System.err.print("Node " + nxtNd
							+ " isn't existed in nodeDict! or");
					System.err.println("Current node " + curNd
							+ " isn't existed in repMatrix");
					continue;
				}
				// adding time information into tmList and calculating time
				// related features
				double curTm = Double.parseDouble(curInfo[1]);
            	double nxtTm = Double.parseDouble(nextInfo[1]);
            	double tmGap = (nxtTm-curTm)/AlgCons.tmDiv;
            	input.tmFeat = TmFeatExtractor.timeFeatExtractor(curTm, prevTm);
				// input.tmFeat.put(0, Math.log(tmGap+AlgCons.eps));
				// input.tmFeat.put(1, tmGap);

				cell.active(t, input, curNdIdx, acts);

				DoubleMatrix d = cell.dDecode(acts.get("h" + t));
                DoubleMatrix lambda = MatrixFunctions.exp(d);
                double logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
						+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap));
				// p(u|H_i)
				DoubleMatrix haty = cell.yDecode(acts.get("h" + t));
				DoubleMatrix py = Activer.softmax(haty);

				cas_logLkHd += (logft + Math.log(py.get(nxtNdIdx)))/(infos.size() - 1);
				cas_mrr += LossFunction.calcMRR(py, nxtNdIdx)/(infos.size()-1);
				
				prevTm = curTm;
			}
			logLkHd -= cas_logLkHd;
			mrr -= cas_mrr;
		}
		logLkHd /= crsValSeq.size();
		System.out.println("The likelihood in Validation: " + logLkHd);
		FileUtil.writeln(oswLog, "The likelihood in Validation: " + logLkHd);
		mrr /= crsValSeq.size();
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

		return logLkHd;
	}
	
	public static double validationOnLoss(InputNeuron input, RNNCell cell, int nodeSize
							, List<String> crsValSeq, OutputStreamWriter oswLog) {

		double logLkHd = .0;
		double mrr = 0;
		for (String seq : crsValSeq) {
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq, input.nodeDict);
			double cas_logLkHd = 0;
			double cas_mrr = 0;
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				String curNd = curInfo[0];
				int curNdIdx = input.nodeDict.indexOf(curNd);
				String nxtNd = nextInfo[0];
				int nxtNdIdx = input.nodeDict.indexOf(nxtNd);
				if (nxtNdIdx < 0 || curNdIdx < 0) {
					System.err.print("Node " + nxtNd + " isn't existed in nodeDict! or");
					System.err.println("Current node " + curNd + " isn't existed in repMatrix");
					continue;
				}
				// adding time information into tmList and calculating time
				// related features
				double tmGap = (Double.parseDouble(nextInfo[1])-Double.parseDouble(curInfo[1]))/AlgCons.tmDiv;
				input.tmFeat = DoubleMatrix.zeros(1, AlgCons.tmFeatSize);
				// input.tmFeat.put(0, Math.log(tmGap+AlgCons.eps));
				// input.tmFeat.put(1, tmGap);

				cell.active(t, input, curNdIdx, acts);
				
				DoubleMatrix d = cell.dDecode(acts.get("h" + t));
                DoubleMatrix lambda = MatrixFunctions.exp(d);
                double logft = Math.log(lambda.get(nxtNdIdx))+input.w.get(0)*tmGap
						+lambda.sum()/input.w.get(0)*(1-Math.exp(input.w.get(0)*tmGap));
                //p(u|H_i)
                DoubleMatrix haty = cell.yDecode(acts.get("h" + t));
                DoubleMatrix py = Activer.softmax(haty);
    	        
                cas_logLkHd += (logft+Math.log(py.get(nxtNdIdx)))/(infos.size()-1);
                cas_mrr += LossFunction.calcMRR(py, nxtNdIdx)/(infos.size()-1);
			}
			logLkHd -= cas_logLkHd;
			mrr -= cas_mrr;
		}
		logLkHd /= crsValSeq.size();
		mrr /= crsValSeq.size();
		System.out.println("The likelihood in Validation: "+logLkHd);
		FileUtil.writeln(oswLog, "The likelihood in Validation: "+logLkHd);
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

		return logLkHd;
	}
	
	public static double validation(InputNeuron input, RNNCell cell, int nodeSize
							, List<String> crsValSeq, OutputStreamWriter oswLog) {
		
		double rmse = .0;
		for(String seq : crsValSeq) {
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq, input.nodeDict);
            double cas_rmse = 0;
			for (int t=0; t<infos.size()-1; t++) {
				String[] curInfo = infos.get(t).split(",");
            	String[] nextInfo = infos.get(t+1).split(",");
            	//translating string node to node index in repMatrix
            	String curNd = curInfo[0];
            	int curNdIdx = input.nodeDict.indexOf(curNd);
            	String nxtNd = nextInfo[0];
            	int nxtNdIdx = input.nodeDict.indexOf(nxtNd);
            	if(nxtNdIdx<0 || curNdIdx<0) {
            		System.err.print("Node "+nxtNd+" isn't existed in nodeDict! or");
            		System.err.println("Current node "+curNd+" isn't existed in repMatrix");
            		continue;
            	}
            	//adding time information into tmList and calculating time related features
            	double tmGap = (Double.parseDouble(nextInfo[1])-Double.parseDouble(curInfo[1]))/AlgCons.tmDiv;
            	input.tmFeat = DoubleMatrix.zeros(1, AlgCons.tmFeatSize);
//            	input.tmFeat.put(0, Math.log(tmGap+AlgCons.eps));
//            	input.tmFeat.put(1, tmGap);
				
				cell.active(t, input, curNdIdx, acts);

				DoubleMatrix d = cell.dDecode(acts.get("h" + t));
				DoubleMatrix lambda = MatrixFunctions.exp(d);
				
				double predTmGap = 1./lambda.get(nxtNdIdx);
				cas_rmse += Math.pow(tmGap-predTmGap, 2.)/(infos.size()-1);
			}
			rmse += Math.sqrt(cas_rmse);
		}
		rmse /= crsValSeq.size();
		System.out.println("The RMSE of time prediction in Validation: "+rmse);
		FileUtil.writeln(oswLog, "The RMSE of time prediction in Validation: "+rmse);
		
		return rmse;
	}
	
	public static double validationInOtherWay(InputNeuron input, RNNCell cell, int nodeSize
							, List<String> crsValSeq, OutputStreamWriter oswLog) {

		double mrr = .0;
		for (String seq : crsValSeq) {
			Map<String, DoubleMatrix> acts = new HashMap<String, DoubleMatrix>();
			List<String> infos = SeqLoader.getNodesAndTimesFromMeme(seq,input.nodeDict);
			double cas_mrr = 0;
			for (int t = 0; t < infos.size() - 1; t++) {
				String[] curInfo = infos.get(t).split(",");
				String[] nextInfo = infos.get(t + 1).split(",");
				// translating string node to node index in repMatrix
				String curNd = curInfo[0];
				int curNdIdx = input.nodeDict.indexOf(curNd);
				String nxtNd = nextInfo[0];
				int nxtNdIdx = input.nodeDict.indexOf(nxtNd);
				if (nxtNdIdx < 0 || curNdIdx < 0) {
					System.err.print("Node " + nxtNd + " isn't existed in nodeDict! or");
					System.err.println("Current node " + curNd + " isn't existed in repMatrix");
					continue;
				}
				// adding time information into tmList and calculating time
				// related features
				double tmGap = (Double.parseDouble(nextInfo[1]) - Double
						.parseDouble(curInfo[1])) / AlgCons.tmDiv;
				input.tmFeat = DoubleMatrix.zeros(1, AlgCons.tmFeatSize);
				// input.tmFeat.put(0, Math.log(tmGap+AlgCons.eps));
				// input.tmFeat.put(1, tmGap);

				cell.active(t, input, curNdIdx, acts);

				DoubleMatrix d = cell.dDecode(acts.get("h" + t));
				DoubleMatrix haty = cell.yDecode(acts.get("h" + t));
				
				DoubleMatrix lambda = MatrixFunctions.exp(d);
				double logSoftmaxDivPart = Math.log(MatrixFunctions.exp(haty).sum());
				
				DoubleMatrix logft = d.sub(lambda.sum()*tmGap);
				DoubleMatrix logpy = haty.sub(logSoftmaxDivPart);
				
				cas_mrr += LossFunction.calcMRR(logpy.add(logft), nxtNdIdx)/(infos.size()-1);
			}
			mrr -= cas_mrr;
		}
		mrr /= crsValSeq.size();
		System.out.println("The MRR of node prediction in Validation: " + mrr);
		FileUtil.writeln(oswLog, "The MRR of node prediction in Validation: " + mrr);

		return mrr;
	}
}
