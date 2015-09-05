#include "ICM.h"

ImageDenoisingProcessor::ImageDenoisingProcessor(double h, double beta, double eta):
                                                h(h), beta(beta), eta(eta)
{
/*************************************************************************************/
//
// Function:
//          Read the image and convert to the binary image;
//          Reset the matrix by replace the '0' with '1'
//
// Parameters:
//          h, beta, eta are all parameters used in the energy function;
//          Adajacent Matrix tells the neighbor of the pixel(i, j);
//          OriginalImage is the image read from .jpg/.png;
//          DenoiseImage is the matrix that we actually operate on;
//          Energy starts from probability respect to tells the joint probbaility of
//          the original image and the denoised image
//
/*************************************************************************************/
    
    cv::Mat iOriginalImage, iBinaryImage;
    iOriginalImage = cv::imread("/Users/liuyang/Desktop/Class/ImageProcessing/ImageDenoising_MRF/Img/flipped.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::threshold(iOriginalImage, iBinaryImage, THRESHOLD, 1, CV_THRESH_BINARY);
    
    imageHeight = iBinaryImage.size[0];
    imageWidth = iBinaryImage.size[1];
    energy = 0;
    
    OriginalImage = ::MatrixXd(imageHeight, imageWidth);
    for (int i = 0; i < imageHeight; i++)
    {
        for (int j = 0; j < imageWidth; j++)
        {
            // Set the Binary value {-1, 1}
            OriginalImage(i, j) = 2 * int(iBinaryImage.at<uchar>(i, j)) - 1;
        }
    }
    DenoisedImage = OriginalImage;
    
    AdjacentMatrix = ::MatrixXd(4, 2);
    AdjacentMatrix << 0, 1,
                    0, -1,
                    1, 0,
                    -1, 0;
}

void ImageDenoisingProcessor::IteratedConditionalModes()
{
/*************************************************************************************/
//
// Function:
//          Localize the energy function iteratively until converge to the local minimum
//          ICM doesn't always converge to the global minimum
//
// Formula:
//          Until converge
//          {
//              mess the index order and iteratively LocalizeEnergy
//          }
//
/*************************************************************************************/

    CalcEnergy();
    while (1)
    {
        double LastEnergy = energy;
        std::vector<int> sHeight, sWidth;

        sHeight = RandSet(imageHeight);
        sWidth = RandSet(imageWidth);
        
        for (auto i = sHeight.cbegin(); i != sHeight.cend(); i++)
        {
            for (auto j = sWidth.cbegin(); j != sWidth.cend(); j++)
            {
                double newEnergy = LocalizeEnergy(*i, *j, energy);
                if (newEnergy < energy)
                {
                    DenoisedImage(*i, *j) = -DenoisedImage(*i, *j);
                    energy = newEnergy;
                }
            }
        }

        if (LastEnergy - energy < 1e-3)
        {
            break;
        }
    }
}

void ImageDenoisingProcessor::CalcEnergy()
{

/*************************************************************************************/
//
// Function:
//          Generate Energy function
// Formula:
//          Energy = h * \sum{x_i} - beta * \sum{x_i x_j} - eta * \sum{x_i y_i}
//
/*************************************************************************************/

    for (long i = 0; i < imageHeight; i++)
    {
        for (long j = 0; j < imageWidth; j++)
        {
            // If the Index out of the range
            MatrixXd neighbor(4, 2);
            neighbor << i, j,
                        i, j,
                        i, j,
                        i, j;
            neighbor = neighbor + AdjacentMatrix;
            neighbor = KickOutlier(neighbor);
            for (long k = 0; k < neighbor.rows(); k++)
            {
                energy -= beta * DenoisedImage(i, j) * DenoisedImage(neighbor(k, 0), neighbor(k, 1));
            }
            energy += h * DenoisedImage(i, j);
            energy -= eta * DenoisedImage(i, j) * OriginalImage(i, j);
        }
    }
}

MatrixXd ImageDenoisingProcessor::KickOutlier(::MatrixXd neighbor) const
{
/*************************************************************************************/
//
// Function:
//          To test if the neighbor coordinates x, y are valid
// Formula:
//          if x > height || x < 0
//              KickOut
//          if y > width || y < 0
//              KickOut
//
/*************************************************************************************/

    long num = neighbor.rows();
    int numOfOut = 0;
    for (int i = 0; i < num; i++)
    {
        if (neighbor(i, 0) >= imageHeight || neighbor(i, 0) < 0)
        {
            numOfOut++;
            continue;
        }
        if (neighbor(i, 1) >= imageWidth || neighbor(i, 1) < 0)
        {
            numOfOut++;
        }
    }

    MatrixXd m(num - numOfOut, 2);
    int index = 0;
    for (int i = 0; i < num; i++)
    {
        if (neighbor(i, 0) < imageHeight && neighbor(i, 0) >= 0)
        {
            if (neighbor(i, 1) < imageWidth && neighbor(i, 1) >= 0)
            {
                m(index, 0) = neighbor(i, 0);
                m(index, 1) = neighbor(i, 1);
                index++;
            }
        }
    }
    
    return m;
}

double ImageDenoisingProcessor::LocalizeEnergy(const long heightIndex, const long widthIndex, double prevEnergy)
{
/*************************************************************************************/
//
// Function:
//          Locally minmize the energy function by keep all pixel fixed except for
//          minimize pixel(i, j)
// Formula:
//          NewEnergy = PreviousEnergy - 2h * x(i, j) + 2eta * x(i, j) * y(i, j) +
//                      2beta * x(i, j) * sum_{x(neighbor_i, neighbor_j)}
//          If NewEnergy < PreviousEnergy
//              Update Energy and change pixel(i, j) to its opposite
//
/*************************************************************************************/
    
    MatrixXd neighbor(4, 2);
    
    neighbor << 1 + heightIndex, widthIndex,
                -1 + heightIndex, widthIndex,
                heightIndex, 1 + widthIndex,
                heightIndex, -1 + widthIndex;
    
    neighbor = KickOutlier(neighbor);
    
    double newEnergy = prevEnergy - 2 * h * DenoisedImage(heightIndex, widthIndex) +
                    2 * eta * DenoisedImage(heightIndex, widthIndex) * OriginalImage(heightIndex, widthIndex);
    
    for (int k = 0; k < neighbor.rows(); k++)
    {
        newEnergy = newEnergy +
                2 * beta * DenoisedImage(heightIndex, widthIndex) * DenoisedImage(neighbor(k, 0), neighbor(k, 1));
    }
    return newEnergy;
}

void ImageDenoisingProcessor::RecoverImage() const
{
/*************************************************************************************/
//
// Function:
//          Recover the {-1, 1} Matrix back to {0, 1} Image
//
/*************************************************************************************/
    cv::Mat iDenoisedImage(int(imageHeight), int(imageWidth), CV_8UC1);
    cv::Mat iOriginalImage(iDenoisedImage.size(), CV_8UC1);
    
    // Recover the image from {-1, 1} to {0, 1}
    for (int i = 0; i < imageHeight; i++)
    {
        for (int j = 0; j < imageWidth; j++)
        {
            iDenoisedImage.at<uchar>(i, j) = uchar((DenoisedImage(i, j) + 1) / 2);
            iOriginalImage.at<uchar>(i, j) = uchar((OriginalImage(i, j) + 1) / 2);
        }
    }
    iDenoisedImage = iDenoisedImage * 255;
    iOriginalImage = iOriginalImage * 255;
    
    cv::imshow("Original", iOriginalImage);
    cv::waitKey();
    cv::imshow("Denoised", iDenoisedImage);
    cv::waitKey();
    
}

std::vector<int> RandSet(long MaxValue)
{
/*************************************************************************************/
//
// Function:
//          Generate a mass order from 1 to MaxValue
//
/*************************************************************************************/
    
    std::set<int> C;
    std::vector<int> S;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> dis(0, int(MaxValue) - 1);
    
    while (C.size() < MaxValue)
    {
        int tmp = dis(generator);
        long SetSize = C.size();
        C.insert(tmp);
        if (C.size() != SetSize)
        {
            S.push_back(tmp);
        }
    }
    return S;
}
