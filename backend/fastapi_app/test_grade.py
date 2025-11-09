import os
from ai_utils import grade_submissions_for_assignment  # your main script file

# ------------------------------
# 1. Configure environment
# ------------------------------
# Make sure these environment variables are set.
# You can either set them here manually or in your environment before running.

# os.environ["GEMINI_API_KEY"] = "<YOUR_GEMINI_API_KEY>"
# os.environ["SUPABASE_URL"] = "<YOUR_SUPABASE_URL>"
# os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "<YOUR_SUPABASE_SERVICE_ROLE_KEY>"

# ------------------------------
# 2. Define test parameters
# ------------------------------
ASSIGNMENT_ID = "0a54f3a4-0e7e-47d1-9054-2058a9b8ccd5"

# This is your rubric — the same format you mentioned earlier.
RUBRIC_TEXT = """
1.a) 0; Not related to question or formula missing,
     2; Mentions averaging but no derivation,
     4; Derivation mostly correct but unclear or missing one key algebraic step,
     5; Fully correct derivation with clear explanation and notation.

2.a) 0; Not related to ridge regression,
     2; Correct OLS part but missing λ regularization term or matrix form errors,
     4; Mostly correct with minor notational or dimension issues,
     5; Correct full objective in compact matrix form, clearly explained.

2.b) 0; Not related or incorrect derivative,
     2; Partial derivative attempt but missing matrix calculus steps,
     4; Mostly correct gradient with small algebraic or sign error,
     5; Correct full gradient derivation in matrix form.

2.c) 0; No relevant attempt,
     2; Incorrect expression or missing normalization term (Nλ),
     4; Almost correct but missing term or unclear algebra,
     5; Fully correct solution (XᵀX + NλI)⁻¹XᵀY.

3.a) 0; No function or incorrect output,
     2; Syntax or logical errors,
     4; Works but inefficient or unclear,
     5; Correct prediction f(x) = wᵀx + b, passes sanity tests.

3.b) 0; Missing or incorrect,
     2; Partial code, does not match formula,
     4; Mostly correct but has minor bug (e.g., bias handling),
     5; Correct and numerically stable implementation.

3.c) 0; No comparison,
     2; MSEs reported but incorrect,
     4; MSEs nearly match but missing clear statement,
     5; Proper comparison and matching MSEs with sklearn OLS.

3.d) 0; No test or wrong setup,
     2; Missing one of train/val/test MSEs,
     4; MSEs computed but mismatch unexplained,
     5; Clear and correct verification that large λ approximates constant regressor.

3.e) 0; No plot,
     2; Incorrect axes or missing labels,
     4; Mostly correct but unclear legend or scaling,
     5; Correct semilogx plot with labeled axes and clear legend.

3.f) 0; Not reported,
     2; Incorrect λ or missing MSEs,
     4; λ and MSEs correct but reasoning unclear,
     5; Correct optimal λ and corresponding train/val/test MSEs, clearly explained.

4.a) 0; No plot,
     2; Partial output or wrong axes,
     4; Plot okay but missing labels or legend,
     5; Correct semilogx plot with 8 lines, labeled and clear.

4.b) 0; No plot,
     2; Plot incomplete or mislabeled,
     4; Minor plotting issues,
     5; Correct semilogx plot matching Ridge format.

4.c) 0; Not answered,
     2; Mentions sparsity vaguely,
     4; Identifies Lasso drives weights to zero but lacks detail,
     5; Correctly explains key difference — Ridge shrinks continuously, Lasso drives some coefficients to zero (sparse solution).

5.a) 0; Not done,
     2; Split done incorrectly or wrong proportions,
     4; Correct split but unclear sizes,
     5; Correct split (80/20) and sizes reported clearly.

5.b) 0; Missing or wrong logic,
     2; Random sampling incorrect or no replacement,
     4; Works but missing “not included %”,
     5; Correct sampling with replacement and accurate exclusion percentage (~36.8%).

5.c) 0; Not implemented,
     2; Partial loop or wrong model usage,
     4; Works but not storing models properly,
     5; Correct ensemble training using K trees.

5.d) 0; Not implemented,
     2; Single-tree prediction only,
     4; Averaging incorrect or incomplete,
     5; Correctly averages predictions across K models.

5.e) 0; No plot,
     2; Missing axis or incomplete legend,
     4; Plot fine but not clearly labeled,
     5; Properly labeled plot showing MSE vs ensemble size.

5.f) 0; No discussion,
     2; Mentions overfitting vaguely,
     4; Partial explanation,
     5; Clearly explains improvement with K and why overfitting does not worsen with bagging.

6.a) 0; Blank,
     0; Short, honest statement provided.

"""

# ------------------------------
# 3. Run the grading pipeline
# ------------------------------
if __name__ == "__main__":
    print("🚀 Starting end-to-end grading test...")

    try:
        results = grade_submissions_for_assignment(
            assignment_id=ASSIGNMENT_ID,
            assignment_idea=RUBRIC_TEXT
        )

        print("\n✅ Finished grading!")
        print(f"Processed {results['count']} submissions.\n")

        for res in results["results"]:
            print(f"Submission ID: {res.get('submission_id')}")
            print(f"User ID: {res.get('user_id')}")
            print(f"Status: {res.get('status')}")
            print("----")

            if res.get("status") == "graded":
                print("Grading Output:\n", res.get("grading"))  # show first 1000 chars
                print("----\n")

    except Exception as e:
        print(f"❌ Error during grading: {e}")
