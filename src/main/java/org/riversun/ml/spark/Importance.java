package org.riversun.ml.spark;

/**
 * Importance of each feature
 * 
 * @author Tom Misawa (riversun.org@gmail.com)
 *
 */
public class Importance {
    public int rawIdx;
    /**
     * Ranking of importance started from 0
     */
    public int rank;
    
    /**
     * col name of feature
     */
    public String name;
    
    /**
     * value of importance 0-1
     */
    public double score;

    /**
     * 
     * @param rawIdx
     * @param name colmn name of feature
     * @param score value of importance 0-1
     * @param rank ranking of importance started from 0
     */
    public Importance(int rawIdx, String name, double score, int rank) {
        this.rawIdx = rawIdx;
        this.name = name;
        this.rank = rank;
        this.score = score;
    }

    @Override
    public String toString() {
        return "FeatureInfo [rank=" + rank + ", score=" + score + ", name=" + name + "]";
    }

}