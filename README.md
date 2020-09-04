LbEnhanced is a new lower bound distance measure that takes advantage of the DTW boundary condition, monotonicity and continuity constraints to create a tighter lower bound. Of particular significance, they remain relatively tight even for large windows. A single parameter to these new lower bounds controls the speed-tightness trade-off.

To read the whole article: https://arxiv.org/pdf/1808.09617v1.pdf

Check out the original implementation, developed by the authors of this new lower bound DTW measure: https://github.com/ChangWeiTan/LbEnhanced.git

What we did is to implement the LbEnhanced measure in Python, based on the implementation made by the authors in Java. Then we integrated the distance measure into a KNN technique. The KNN original implementation is here: https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping.git





