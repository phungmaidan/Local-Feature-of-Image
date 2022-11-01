#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

Mat detectHarris(Mat src, int k, int thresh)
{
    if (!src.data)
        return Mat();
    Mat dstImage = src.clone(), srcGrayscale;

    cvtColor(src, srcGrayscale, COLOR_BGR2GRAY);
    Mat dst = Mat::zeros(src.size(), CV_32FC1);

    Mat Ix, Iy, trace, det;
    Mat Ixx, Iyy, Ixy, Ixx_gaus, Iyy_gaus, Ixy_gaus;

    //Tính đạo hàm theo x,y bằng hàm Sobel, I_y, I_y
    Sobel(srcGrayscale, Ix, CV_32FC1, 1, 0, 3, BORDER_DEFAULT);
    Sobel(srcGrayscale, Iy, CV_32FC1, 0, 1, 3, BORDER_DEFAULT);

    //Tính Ix^2, Iy^2 và Ixy
    pow(Ix, 2.0, Ixx);
    pow(Iy, 2.0, Iyy);
    multiply(Ix, Iy, Ixy);

    //Áp dụng Gaussian làm trơn ảnh
    GaussianBlur(Ixx, Ixx_gaus, Size(3, 3), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(Iyy, Iyy_gaus, Size(3, 3), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy, Ixy_gaus, Size(3, 3), 2.0, 2.0, BORDER_DEFAULT);

    //Tính corner response theo công thức R = det(M) - k * trace(M)
    //Tính det:
    Mat temp1, temp2;
    multiply(Ixx_gaus, Iyy_gaus, temp1);
    multiply(Ixy_gaus, Ixy_gaus, temp2);
    det = temp1 - temp2;
    //Tính trace:
    pow((Ixx_gaus + Iyy_gaus), 2.0, trace);
    //Tính R = dst (k là giá trị thực nghiệm)
    dst = det - k * trace;

    //Đánh dấu và khoanh tròn các điểm đặc trưng
    Mat dstNorm, dstNormScaled;
    normalize(dst, dstNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if ((int)dstNorm.at<float>(i, j) > thresh)
            {
                circle(dstImage, Point(j, i), 3, Scalar(0), 2, 8, 0);
            }
        }
    }
    return dstImage;
}
int main(int argc, char** argv)
{

    if (argc < 3)
    {
        cout << "Chuong trinh khong the thuc hien" << endl;
        return -1;
    }
    Mat src;
    Mat dstImage;
    src = imread(argv[1], IMREAD_COLOR);
    if (!src.data)
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    string* str = new string[argc - 1];
    for (int i = 1; i < argc; i++)
    {
        str[i - 1] = argv[i];
        cout << str[i - 1] << endl;
    }
    namedWindow("Source image");
    imshow("Source image", src);
    if (str[1] == "detectHarris")   
    {
        double k = atof(str[2].c_str());
        int thresh = atoi(str[3].c_str());
        if (thresh > 255 || thresh < 0)
        {
            cout << "Gia tri thresh nhap khong hop le!!!\n";
            return -1;
        }
        dstImage = detectHarris(src, k, thresh);
        namedWindow("Dst image");
        imshow("Dst image", dstImage);
    }

    waitKey(0);
    return 0;
}