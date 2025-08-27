import json
import os
from typing import Dict, List, Any

def get_difficulty_grade_corrected(score: int) -> str:
    """Convert visual difficulty score to MCP difficulty grade using correct boundaries"""
    if 81 <= score <= 100:
        return "easy"
    elif 61 <= score <= 80:
        return "moderate" 
    elif 41 <= score <= 60:
        return "difficult"
    elif 21 <= score <= 40:
        return "very_difficult"
    elif 1 <= score <= 20:
        return "extremely_difficult"
    else:
        return "unknown"

def load_difficulty_ratings(ratings_file: str) -> Dict[str, Dict]:
    """Load visual difficulty ratings for all images with corrected grading"""
    with open(ratings_file, 'r') as f:
        data = json.load(f)
    
    # Create lookup dictionary by image ID
    ratings_lookup = {}
    for result in data["detailed_results"]:
        img_id = result["img_id"]
        score = result["visual_difficulty_score"]
        corrected_grade = get_difficulty_grade_corrected(score)
        
        ratings_lookup[img_id] = {
            "difficulty_grade": corrected_grade,
            "visual_difficulty_score": score,
            "features": result.get("features", {}),
            "ease_indicators": result.get("ease_indicators", [])
        }
    
    return ratings_lookup

def calculate_average_distance(results: List[Dict]) -> Dict:
    """Calculate average distance metrics from test results"""
    distances = []
    
    # Extract distances from results
    for result in results:
        dist = result.get("distance_km")
        if dist is not None and dist != "N/A" and isinstance(dist, (int, float)):
            distances.append(dist)
    
    if not distances:
        return {
            "count": 0,
            "average": None,
            "median": None,
            "min": None,
            "max": None
        }
    
    distances.sort()
    n = len(distances)
    
    return {
        "count": n,
        "average": sum(distances) / n,
        "median": distances[n // 2] if n % 2 == 1 else (distances[n // 2 - 1] + distances[n // 2]) / 2,
        "min": distances[0],
        "max": distances[-1]
    }

def analyze_by_difficulty_level_corrected(test_results, difficulty_ratings: Dict[str, Dict]) -> Dict:
    """Analyze test results with proper image ID cross-referencing"""
    
    # Group results by difficulty level
    difficulty_groups = {
        "easy": [],
        "moderate": [], 
        "difficult": [],
        "very_difficult": [],
        "extremely_difficult": []
    }
    
    # Handle both list and dict formats
    if isinstance(test_results, list):
        detailed_results = test_results
    else:
        detailed_results = test_results.get("detailed_results", [])
    
    # Cross-reference using image IDs
    matched_count = 0
    unmatched_images = []
    
    for result in detailed_results:
        img_id = result.get("img_id", "")
        
        if img_id in difficulty_ratings:
            difficulty_grade = difficulty_ratings[img_id]["difficulty_grade"]
            result["difficulty_info"] = difficulty_ratings[img_id]
            difficulty_groups[difficulty_grade].append(result)
            matched_count += 1
        else:
            unmatched_images.append(img_id)
    
    print(f"Matched {matched_count}/{len(detailed_results)} images with difficulty ratings")
    if unmatched_images:
        print(f"First 5 unmatched images: {unmatched_images[:5]}")
    
    # Calculate metrics for each difficulty level
    difficulty_analysis = {}
    
    for difficulty, results in difficulty_groups.items():
        if not results:
            continue
            
        total_images = len(results)
        
        # For Mode 1/2 format - check if prediction is not "unknown"
        if any("prediction" in r for r in results):
            successful_predictions = len([r for r in results 
                                        if (r.get("prediction") is not None and 
                                            ((r.get("prediction") or {}).get("country") or "").lower() != "unknown")])
        # For comparison format
        elif any("mode1_result" in r or "mode7_result" in r for r in results):
            successful_predictions = len([r for r in results 
                                        if (r.get("mode1_result", {}).get("success", False) or 
                                            r.get("mode7_result", {}).get("success", False))])
        else:
            successful_predictions = len([r for r in results if r.get("success", False)])
        
        # Calculate accuracy over ALL images in this difficulty level (not just successful ones)
        if any("matches" in r for r in results):
            # Mode 1/2 format with pre-calculated matches
            country_accuracy = sum(1 for r in results 
                                 if r.get("matches", {}).get("country", False)) / total_images
            state_accuracy = sum(1 for r in results 
                               if r.get("matches", {}).get("state", False)) / total_images
            city_accuracy = sum(1 for r in results 
                              if r.get("matches", {}).get("city", False)) / total_images
        elif any("comparison" in r for r in results):
            # Comparison format
            country_accuracy = sum(1 for r in results 
                                 if r.get("comparison", {}).get("country_match", False)) / total_images
            state_accuracy = sum(1 for r in results 
                               if r.get("comparison", {}).get("state_match", False)) / total_images
            city_accuracy = sum(1 for r in results 
                              if r.get("comparison", {}).get("city_match", False)) / total_images
        else:
            # Manual calculation for other formats
            country_matches = 0
            state_matches = 0 
            city_matches = 0
            
            for r in results:
                gt = r.get("ground_truth", {})
                
                # Check different prediction formats
                pred = None
                if "prediction" in r:
                    pred = r["prediction"] if r["prediction"] is not None else {}
                elif "mode1_result" in r and r["mode1_result"].get("success", False):
                    pred = r["mode1_result"].get("prediction", {})
                elif "mode7_result" in r and r["mode7_result"].get("success", False):
                    pred = r["mode7_result"].get("prediction", {})
                
                if pred and isinstance(pred, dict):
                    # Case-insensitive matching, excluding "unknown"
                    if (gt.get("country", "").lower() == pred.get("country", "").lower() and 
                        gt.get("country", "") != "" and pred.get("country", "").lower() != "unknown"):
                        country_matches += 1
                    if (gt.get("state", "").lower() == pred.get("state_region", "").lower() and 
                        gt.get("state", "") != "" and pred.get("state_region", "").lower() != "unknown"):
                        state_matches += 1
                    if (gt.get("city", "").lower() == pred.get("city", "").lower() and 
                        gt.get("city", "") != "" and pred.get("city", "").lower() != "unknown"):
                        city_matches += 1
            
            country_accuracy = country_matches / total_images
            state_accuracy = state_matches / total_images
            city_accuracy = city_matches / total_images
        
        # Calculate average processing time
        processing_times = []
        for r in results:
            pt = r.get("processing_time", 0) or r.get("processing_time_seconds", 0)
            processing_times.append(pt)
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate distance metrics (if available)
        distances = []
        for r in results:
            dist = r.get("distance_km")
            if dist is not None and dist != "N/A" and isinstance(dist, (int, float)):
                distances.append(dist)
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            median_distance = sorted(distances)[len(distances)//2]
            min_distance = min(distances)
            max_distance = max(distances)
            
            # Calculate distance distribution
            total_with_distance = len(distances)
            under_5km = sum(1 for d in distances if d < 5)
            under_50km = sum(1 for d in distances if d < 50)
            under_500km = sum(1 for d in distances if d < 500)
            under_1000km = sum(1 for d in distances if d < 1000)
            
            distance_distribution = {
                "under_5km_count": under_5km,
                "under_50km_count": under_50km,
                "under_500km_count": under_500km,
                "under_1000km_count": under_1000km,
                "under_5km_percent": (under_5km / total_with_distance * 100) if total_with_distance > 0 else 0,
                "under_50km_percent": (under_50km / total_with_distance * 100) if total_with_distance > 0 else 0,
                "under_500km_percent": (under_500km / total_with_distance * 100) if total_with_distance > 0 else 0,
                "under_1000km_percent": (under_1000km / total_with_distance * 100) if total_with_distance > 0 else 0,
                "total_with_distance": total_with_distance
            }
        else:
            avg_distance = median_distance = min_distance = max_distance = None
            distance_distribution = {
                "under_5km_count": 0,
                "under_50km_count": 0,
                "under_500km_count": 0,
                "under_1000km_count": 0,
                "under_5km_percent": 0,
                "under_50km_percent": 0,
                "under_500km_percent": 0,
                "under_1000km_percent": 0,
                "total_with_distance": 0
            }
        
        difficulty_analysis[difficulty] = {
            "total_images": total_images,
            "successful_predictions": successful_predictions,
            "success_rate": successful_predictions / total_images if total_images > 0 else 0,
            "accuracy_metrics": {
                "country_accuracy": country_accuracy,
                "state_region_accuracy": state_accuracy,
                "city_accuracy": city_accuracy
            },
            "performance_metrics": {
                "avg_processing_time_seconds": avg_processing_time,
                "avg_distance_error_km": avg_distance,
                "median_distance_error_km": median_distance,
                "min_distance_error_km": min_distance,
                "max_distance_error_km": max_distance
            },
            "distance_distribution": distance_distribution
        }
    
    return difficulty_analysis

def enhance_test_results_corrected(test_file: str, difficulty_file: str, output_file: str):
    """Enhanced test results with corrected image ID cross-referencing"""
    
    # Load existing test results
    with open(test_file, 'r') as f:
        test_results = json.load(f)
    
    # Load difficulty ratings
    difficulty_ratings = load_difficulty_ratings(difficulty_file)
    
    print(f"Loaded {len(difficulty_ratings)} difficulty ratings")
    
    # Perform corrected difficulty-based analysis
    difficulty_analysis = analyze_by_difficulty_level_corrected(test_results, difficulty_ratings)
    
    # Create enhanced results structure
    if isinstance(test_results, list):
        enhanced_results = {
            "original_results": test_results,
            "difficulty_level_analysis": difficulty_analysis,
            "summary": {
                "total_images": len(test_results),
                "analysis_type": "corrected_difficulty_analysis"
            }
        }
    else:
        enhanced_results = test_results.copy()
        enhanced_results["difficulty_level_analysis"] = difficulty_analysis
    
    # Calculate overall difficulty distribution
    total_analyzed = sum(analysis["total_images"] for analysis in difficulty_analysis.values())
    difficulty_distribution = {}
    for difficulty, analysis in difficulty_analysis.items():
        count = analysis["total_images"]
        percentage = (count / total_analyzed * 100) if total_analyzed > 0 else 0
        difficulty_distribution[difficulty] = {
            "count": count,
            "percentage": percentage
        }
    
    enhanced_results["difficulty_distribution"] = difficulty_distribution
    
    # Save enhanced results
    with open(output_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("=" * 70)
    print("CORRECTED DIFFICULTY-BASED ANALYSIS")
    print("=" * 70)
    
    if isinstance(test_results, list):
        print(f"\nTest Mode: Comparison Results")
    else:
        print(f"\nTest Mode: {test_results.get('test_summary', {}).get('test_mode', 'Unknown')}")
    print(f"Total Images Analyzed: {total_analyzed}")
    
    print("\nDifficulty Level Distribution:")
    print("-" * 40)
    for difficulty, dist in difficulty_distribution.items():
        print(f"{difficulty:>18}: {dist['count']:>3} images ({dist['percentage']:>5.1f}%)")
    
    print("\nPerformance by Difficulty Level:")
    print("-" * 100)
    print(f"{'Level':<18} {'Images':<8} {'Success':<8} {'Country':<8} {'State':<8} {'City':<8} {'Avg Time':<10} {'Avg Dist':<10}")
    print("-" * 100)
    
    for difficulty, analysis in difficulty_analysis.items():
        if analysis["total_images"] > 0:
            avg_dist = analysis['performance_metrics']['avg_distance_error_km']
            dist_str = f"{avg_dist:.1f}" if avg_dist is not None else "N/A"
            print(f"{difficulty:<18} {analysis['total_images']:<8} "
                  f"{analysis['success_rate']:<8.1%} "
                  f"{analysis['accuracy_metrics']['country_accuracy']:<8.1%} "
                  f"{analysis['accuracy_metrics']['state_region_accuracy']:<8.1%} "
                  f"{analysis['accuracy_metrics']['city_accuracy']:<8.1%} "
                  f"{analysis['performance_metrics']['avg_processing_time_seconds']:<10.1f} "
                  f"{dist_str:<10}")
    
    print("\nDistance Statistics by Difficulty Level:")
    print("-" * 80)
    print(f"{'Level':<18} {'Avg (km)':<10} {'Median (km)':<12} {'Min (km)':<10} {'Max (km)':<10}")
    print("-" * 80)
    
    for difficulty, analysis in difficulty_analysis.items():
        if analysis["total_images"] > 0:
            metrics = analysis['performance_metrics']
            avg_dist = f"{metrics['avg_distance_error_km']:.1f}" if metrics['avg_distance_error_km'] is not None else "N/A"
            median_dist = f"{metrics['median_distance_error_km']:.1f}" if metrics['median_distance_error_km'] is not None else "N/A"
            min_dist = f"{metrics['min_distance_error_km']:.1f}" if metrics['min_distance_error_km'] is not None else "N/A"
            max_dist = f"{metrics['max_distance_error_km']:.1f}" if metrics['max_distance_error_km'] is not None else "N/A"
            
            print(f"{difficulty:<18} {avg_dist:<10} {median_dist:<12} {min_dist:<10} {max_dist:<10}")
    
    print("\nDistance Distribution by Difficulty Level:")
    print("-" * 90)
    print(f"{'Level':<18} {'<5km':<12} {'<50km':<12} {'<500km':<12} {'<1000km':<12} {'Total':<8}")
    print("-" * 90)
    
    for difficulty, analysis in difficulty_analysis.items():
        if analysis["total_images"] > 0 and analysis["distance_distribution"]["total_with_distance"] > 0:
            dist_data = analysis["distance_distribution"]
            print(f"{difficulty:<18} "
                  f"{dist_data['under_5km_count']:>3}({dist_data['under_5km_percent']:>4.1f}%) "
                  f"{dist_data['under_50km_count']:>3}({dist_data['under_50km_percent']:>4.1f}%) "
                  f"{dist_data['under_500km_count']:>3}({dist_data['under_500km_percent']:>4.1f}%) "
                  f"{dist_data['under_1000km_count']:>3}({dist_data['under_1000km_percent']:>4.1f}%) "
                  f"{dist_data['total_with_distance']:<8}")
    
    print(f"\nCorrected results saved to: {output_file}")

def main():
    """Main function to run corrected difficulty analysis"""
    
    # File paths
    difficulty_ratings_file = "visual_difficulty_ratings_mcp_corrected.json"
    
    # Test files to enhance
    test_files = [
#        "test_output_doxbench_gemini_25_pro/test_report.json",
#        "test_output_doxbench_gemini_25_pro_7/test_report.json",
#        "test_output_doxbench_gemini_25_flash/test_report.json",
#        "test_output_doxbench_gemini_25_flash_7/test_report.json",
#        "test_output_doxbench_o3/test_report.json",
#        "test_output_doxbench_o3_7/test_report.json",
#        "test_output_doxbench_4o/test_report.json",
#        "test_output_doxbench_4o_7/test_report.json",
#        "test_output_gemini_pro_7/test_report.json",
#        "test_output_gemini_pro_1/test_report.json",
         "test_output_7_o3/test_report.json",
         "test_output_o3/test_report.json",
         #"test_output_mode1/test_report.json",
         #"test_output_mode777/test_report.json",

    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            output_file = test_file.replace('.json', '_corrected_difficulty.json')
            print(f"\nProcessing: {test_file}")
            enhance_test_results_corrected(test_file, difficulty_ratings_file, output_file)
        else:
            print(f"File not found: {test_file}")

if __name__ == "__main__":
    main()
