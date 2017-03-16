package com.kingwang.rnntd.dataset;

import java.util.ArrayList;
import java.util.List;

public class SeqLoader {

	/**
	 * Get nodes from memetracker format file and filter nodes that doesn't exist in nodeSet 
	 * 
	 * @param seq
	 * @param nodeSet
	 * @return
	 */
	public static List<String> getNodesFromMeme(String seq, List<String> nodeSet) {
    	
		List<String> nodes = new ArrayList<>();
		
    	String[] elems = seq.split(",");
    	for(int i=1; i<elems.length; i+=2) {
    		if(nodeSet.contains(elems[i])) {
    			nodes.add(elems[i]);
    		}
    	}
    	
    	return nodes;
    }
	
	public static List<String> getNodesAndTimesFromMeme(String seq, List<String> nodeSet) {
		
		List<String> ndsAndTms = new ArrayList<>();
		
    	String[] elems = seq.split(",");
    	for(int i=1; i<elems.length; i+=2) {
    		if(i+1>=elems.length) {
    			break;
    		}
    		String info = "";
    		if(nodeSet.contains(elems[i])) {
    			info = elems[i]+","+elems[i+1];
    			ndsAndTms.add(info);
    		} 
    	}
    	
    	return ndsAndTms;
	}
}
