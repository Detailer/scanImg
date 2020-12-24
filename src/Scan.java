import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import java.lang.Object;

public class Scan{
    static void viewImage(String extension, Mat img) throws IOException{
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(extension, img, matOfByte);

        byte[] byteArray = matOfByte.toArray();

        InputStream in = new ByteArrayInputStream(byteArray);
        BufferedImage bufferedImage = ImageIO.read(in);

        // Insert Buffered Image in frame and Display
        JFrame frame = new JFrame();
        frame.getContentPane().add(new JLabel(new ImageIcon(bufferedImage)));
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String args[]) throws IOException{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String File = "./src/img/image-2.jpg";
        Mat orig = Imgcodecs.imread(File);

        Mat src = new Mat();
        Size newsz = new Size(0, 0);
        System.out.println(orig.size().width);
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

        //Collections.sort(contours, Collections.reverseOrder());
        List<MatOfPoint> rect_contour= new ArrayList<>();
        //List<Point> rect_points = new ArrayList<>();
        boolean found = false;
        Point[] sortedPoints = new Point[4];
        for (MatOfPoint contour: contours) {
            MatOfPoint2f contourFloat = new MatOfPoint2f(contour.toArray());
            double arc = Imgproc.arcLength(contourFloat, true) * 0.02;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contourFloat, approx, arc, true);
            if (approx.total() == 4 && Imgproc.contourArea(contour) > 5000.0) {
                found = true;
                rect_contour.add(contour);
                Scalar color = new Scalar(189, 145, 35);
                Mat srcContor = src.clone();
                Imgproc.drawContours(srcContor, rect_contour, -1, color, Imgproc.LINE_8);
                HighGui.imshow("Contours", srcContor);

                //calculate the center of mass of our contour image using moments
                Moments moment = Imgproc.moments(approx);
                int x = (int) (moment.get_m10() / moment.get_m00());
                int y = (int) (moment.get_m01() / moment.get_m00());

                //SORT POINTS RELATIVE TO CENTER OF MASS
                double[] data;
                int count = 0;
                for(int i = 0; i < approx.total(); i++){
                    data = approx.get(i, 0);
                    double datax = data[0];
                    double datay = data[1];
                    if(datax < x && datay < y){
                        sortedPoints[0] = new Point(datax,datay);
                        count++;
                    }else if(datax > x && datay < y){
                        sortedPoints[1] = new Point(datax,datay);
                        count++;
                    }else if (datax < x && datay > y){
                        sortedPoints[2] = new Point(datax,datay);
                        count++;
                    }else if (datax > x && datay > y){
                        sortedPoints[3] = new Point(datax,datay);
                        count++;
                    }
                }
                /*
                for (int j = 0; j <  4; j++){
                    double[] temp;
                    temp = approx.get(j, 0);
                    rect_points.add(new Point(temp[0], temp[1]));
                    //Imgproc.drawMarker(src, rect_points.get(j), color);
                }*/
                break;
            }
        }
        if (found){
            MatOfPoint2f source = new MatOfPoint2f(
                    sortedPoints[0],
                    sortedPoints[1],
                    sortedPoints[2],
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
            HighGui.imshow("Warpped Image", warpped);

            Mat warrpedGrey = new Mat();
            Imgproc.cvtColor(warpped, warrpedGrey, Imgproc.COLOR_RGB2GRAY);

            Mat finalImg = new Mat();
            Imgproc.adaptiveThreshold(warrpedGrey, finalImg, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

            HighGui.imshow("Final Image", finalImg);
        }
        else {
            HighGui.imshow("Source Image", src);
        }
        HighGui.waitKey(1);

    }
}