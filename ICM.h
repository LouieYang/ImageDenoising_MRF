#ifndef ImageDenoising_MRF_ICM_h
#define ImageDenoising_MRF_ICM_h

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <set>

#include <time.h>
#include <random>

#define THRESHOLD 100
#define BATCHSIZE 1000

using Eigen::MatrixXd;

std::vector<int> RandSet(long MaxValue);

class ImageDenoisingProcessor
{
public:
    using nPixel = long;
    
    ImageDenoisingProcessor(): ImageDenoisingProcessor(0, 1e-3, 2.1e-3) {};
    ImageDenoisingProcessor(double h, double beta, double eta);
    
    void IteratedConditionalModes();
    double LocalizeEnergy(const long heightIndex, const long widthIndex, double prevEnergy);
    void RecoverImage() const;
    
private:
    
    void CalcEnergy();
    ::MatrixXd KickOutlier(::MatrixXd neighbor) const;
    
    double h;
    double beta;
    double eta;
    double energy;
    
    nPixel imageHeight;
    nPixel imageWidth;
    
    ::MatrixXd OriginalImage;
    ::MatrixXd DenoisedImage;
    ::MatrixXd AdjacentMatrix;
};
#endif
