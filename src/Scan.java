import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

import java.util.*;

import java.util.List;

public class Scan{
    public static void main(String[] args) throws IOException{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String File = "./src/img/toScan.png";
        Mat orig = Imgcodecs.imread(File);

        // resizing source
        Mat src = new Mat();
        Size newsz = new Size(0, 0);
        double scale = (float) 500 / orig.size().width;
        Imgproc.resize(orig, src, newsz, scale, scale, Imgproc.INTER_AREA);
        double h = src.size().height;
        double w = src.size().width;
        HighGui.imshow("Resized", src);

        Mat grey = new Mat();
        Imgproc.cvtColor(src, grey, Imgproc.COLOR_RGB2GRAY);

        Mat blur = new Mat();
        Imgproc.GaussianBlur(grey, blur, new Size(5, 5), 0);

        Mat edge = new Mat();
        Imgproc.Canny(grey, edge, 75, 200);

        List<MatOfPoint> contours= new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edge, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("Number of Contours Found: " + contours.size());

        // draw all found contours
        Mat allContours = src.clone();
        Imgproc.drawContours(allContours, contours, -1, new Scalar(0,0, 255 ), Imgproc.LINE_8);
        HighGui.imshow("All Contours",allContours);

        boolean found = false;
        Point[] sortedPoints = new Point[4];
        for (MatOfPoint contour : contours) {
            //approximate the contour
            MatOfPoint2f contourFloat = new MatOfPoint2f(contour.toArray());
            double arc = Imgproc.arcLength(contourFloat, true) * 0.02;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contourFloat, approx, arc, true);

            if (approx.total() == 4 && Imgproc.contourArea(contour) > 1000.0) {
                found = true;

                // display rect. contour
                List<MatOfPoint> rect_contour = new ArrayList<>();
                rect_contour.add(contour);
                Scalar color = new Scalar(189, 145, 35);
                Mat srcContor = src.clone();
                Imgproc.drawContours(srcContor, rect_contour, -1, color, Imgproc.LINE_8);
                HighGui.imshow("Contours", srcContor);

                // store points from approx in array and sort
                for (int i = 0; i < approx.total(); i++) {
                    double[] temp = approx.get(i, 0);
                    double dataX = temp[0];
                    double dataY = temp[1];
                    sortedPoints[i] = new Point(dataX, dataY);
                }
                Arrays.sort(sortedPoints, (a, b) -> {
                    int xComp = Double.compare(a.x, b.x);
                    if (xComp == 0)
                        return Double.compare(a.y, b.y);
                    else
                        return xComp;
                });
                break;
            }
        }
        if (found){
            MatOfPoint2f source = new MatOfPoint2f(
                    sortedPoints[0],
                    sortedPoints[2],
                    sortedPoints[1],
                    sortedPoints[3]
            );
            MatOfPoint2f destination = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(w - 1, 0),
                    new Point(0, h - 1),
                    new Point(w - 1, h - 1)
            );

            Mat warpMat = Imgproc.getPerspectiveTransform(source, destination);
            Mat warpped = new Mat();
            Imgproc.warpPerspective(src, warpped, warpMat, src.size());
            // dsiplay warpped image
            HighGui.imshow("Warpped Image", warpped);

            Mat warrpedGrey = new Mat();
            Imgproc.cvtColor(warpped, warrpedGrey, Imgproc.COLOR_RGB2GRAY);

            Mat finalImg = new Mat();
            Imgproc.adaptiveThreshold(warrpedGrey, finalImg, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

            // display final image
            HighGui.imshow("Final Image", finalImg);
        }
        else {
            // if rect. not found then show original image
            HighGui.imshow("Source Image", src);
        }
        HighGui.waitKey(1);

    }
}