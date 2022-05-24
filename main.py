import argparse
import time
from collect import extract_image
from detect import detect_object

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area", default="all", help="The AREA of which the traffic images \
        will be extracted (e.g., sle, tpe) or 'all' for all areas.")
    parser.add_argument(
        "--output_dir", default="./images/",
        help="The output directory for storing the collected images."
    )
    parser.add_argument(
        "--model_quality", default="medium",
        help="Choose from low, medium, high. A higher quality model takes longer time to detect."
    )
    parser.add_argument(
        "--confidence", default=0.5,
        help="The confidence level to sift predictions with high certainties."
    )
    parser.add_argument(
        "--target_class", default="car",
        help="Target object class for detection."
    )

    # Options:
    # "WL", "SLE", "TPE", "KJE", "BKE", "CTE",
    # "PIE", "KPE", "AYE", "MCE", "ECP", "STG", "ALL"
    args = parser.parse_args()
    areas = ["woodlands", "sle", "tpe", "kje", "bke", "cte",
             "pie", "kpe", "aye", "mce", "ecp", "stg"]

    start_time = time.time()

    # Task 1: Extract Images from the Website
    if args.area == "all":
        for area in areas:
            extract_image(area, main_dir=args.output_dir)
    elif args.area in areas:
        extract_image(args.area, main_dir=args.output_dir)
    else:
        raise TypeError("Invalid area argument.")

    # Task 2: Detect and Count Cars
    if args.area == "all":
        for area in areas:
            detect_object(area,
                          confidence=args.confidence,
                          tar_class=args.target_class,
                          main_dir=args.output_dir,
                          model_quality=args.model_quality)
    elif args.area in areas:
        detect_object(args.area,
                      confidence=args.confidence,
                      tar_class=args.target_class,
                      main_dir=args.output_dir,
                      model_quality=args.model_quality)

    print("Elapsed time = % 0.4fs" % (time.time() - start_time))
    print("Done!")
