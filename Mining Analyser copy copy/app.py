from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tempfile
import PyPDF2
import requests
import json
import logging
from dotenv import load_dotenv
import time
import traceback

# Load environment variables from .env file if present
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 32MB max file size

# Get API key from environment variable
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file with robust error handling"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Check if PDF is empty
            if len(reader.pages) == 0:
                return "Error: The PDF contains no pages."

            # Extract text from each page with page-specific error handling
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logger.warning(f"Page {i + 1} contains no extractable text.")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i + 1}: {str(e)}")
                    # Continue with other pages even if one fails

            # Check if we managed to extract any text at all
            if not text.strip():
                return "Error: Could not extract any text from the PDF. The document might be scanned, protected, or contain only images."

    except PyPDF2.errors.PdfReadError as e:
        error_msg = f"PDF read error: {str(e)}. The file might be corrupted or password-protected."
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(error_msg)
        return error_msg

    # Log statistics about extracted text
    logger.info(f"Successfully extracted {len(text)} characters from PDF")
    return text


def format_analysis_results(analysis_text):
    """Format the raw analysis text into structured HTML with better formatting"""
    # Check if there was an error
    if analysis_text.startswith("Error:"):
        return f'<div class="error-message">{analysis_text}</div>'

    formatted_html = '<div class="analysis-results">'

    # Split the analysis into sections
    lines = analysis_text.split('\n')
    current_section = None
    section_content = []
    ratings = {}

    # First pass: extract ratings for aspects
    for line in lines:
        # Look for rating patterns like "1. Resource Quality & Certainty: 7/10"
        if ':' in line and ('/10' in line or '/ 10' in line):
            parts = line.split(':')
            if len(parts) >= 2:
                aspect_name = parts[0].strip()
                if any(digit in aspect_name for digit in '123456789'):
                    # Extract number from aspect rating
                    rating_text = parts[1].strip()
                    try:
                        # Extract just the number before /10
                        rating_value = int(rating_text.split('/')[0].strip())
                        # Remove any numbers from the aspect name
                        clean_aspect = ''.join([c for c in aspect_name if not c.isdigit()]).strip()
                        # Remove any leading dots or other punctuation
                        clean_aspect = clean_aspect.lstrip('. ')
                        ratings[clean_aspect] = rating_value
                    except (ValueError, IndexError):
                        pass

    # Add a ratings overview section at the top if we found ratings
    if ratings:
        formatted_html += '<div class="ratings-overview">'
        formatted_html += '<h2>Ratings Overview</h2>'
        formatted_html += '<div class="ratings-grid">'

        for aspect, rating in ratings.items():
            # Calculate the percentage for the progress bar
            percentage = (rating / 10) * 100

            # Determine color based on rating
            if rating >= 8:
                color = "#28a745"  # Green for high ratings
            elif rating >= 6:
                color = "#ffc107"  # Yellow for medium ratings
            else:
                color = "#dc3545"  # Red for low ratings

            formatted_html += f'''
            <div class="rating-item">
                <div class="aspect-name">{aspect}</div>
                <div class="rating-bar-container">
                    <div class="rating-bar" style="width: {percentage}%; background-color: {color};"></div>
                </div>
                <div class="rating-value">{rating}/10</div>
            </div>
            '''

        formatted_html += '</div></div>'  # Close ratings-grid and ratings-overview

    # Second pass: build the formatted content
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_html += '</ul>'
                in_list = False
            if current_section:
                formatted_html += f'<div class="section-content">{"".join(section_content)}</div></div>'
                current_section = None
                section_content = []
            continue

        # Check for section headers (identified by ending with a colon)
        if line.endswith(':') and len(line) < 100 and not line.startswith('- '):
            if in_list:
                formatted_html += '</ul>'
                in_list = False
            if current_section:
                formatted_html += f'<div class="section-content">{"".join(section_content)}</div></div>'
                current_section = line[:-1]  # Remove the colon
                section_content = []
                formatted_html += f'<div class="section"><h3>{current_section}</h3>'
            else:
                current_section = line[:-1]  # Remove the colon
                formatted_html += f'<div class="section"><h3>{current_section}</h3>'
                section_content = []

        # Check for main headings like "Key Strengths", "Key Weaknesses", "Recommendations", etc.
        elif any(heading in line.lower() for heading in ["key strengths", "key weaknesses", "recommendations",
                                                         "financing potential", "likelihood", "overall score"]):
            if in_list:
                formatted_html += '</ul>'
                in_list = False
            if current_section:
                formatted_html += f'<div class="section-content">{"".join(section_content)}</div></div>'
                current_section = None
                section_content = []

            formatted_html += f'<div class="major-section"><h2>{line}</h2>'
            current_section = "major"

        # Check for bullet points
        elif line.startswith('- ') or line.startswith('* ') or (
                line.startswith(tuple('123456789')) and '. ' in line[:5]):
            if not in_list:
                if section_content:
                    # Add any previous paragraph text
                    formatted_html += f'<p>{"".join(section_content)}</p>'
                    section_content = []
                formatted_html += '<ul class="bullet-list">'
                in_list = True

            # Extract text after the bullet point
            if line.startswith('- ') or line.startswith('* '):
                bullet_text = line[2:]
            else:
                bullet_text = line.split('. ', 1)[1] if '. ' in line else line

            formatted_html += f'<li>{bullet_text}</li>'

        # Regular content
        else:
            if in_list:
                formatted_html += '</ul>'
                in_list = False

            # Handle numeric ratings within text
            if ':' in line and '/10' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    aspect = parts[0].strip()
                    explanation = parts[1].strip()
                    # Check if this starts with a number
                    if any(aspect.startswith(str(i)) for i in range(1, 10)):
                        # This is a rating line
                        section_content.append(
                            f'<div class="rating-line"><strong>{aspect}:</strong> {explanation}</div>')
                    else:
                        section_content.append(f'<p><strong>{aspect}:</strong> {explanation}</p>')
                else:
                    section_content.append(f'<p>{line}</p>')
            else:
                # Check for "score" mentions for highlighting
                if "score:" in line.lower() or "potential:" in line.lower() or "overall:" in line.lower():
                    # Extract the score value
                    try:
                        score_text = line.split(":")[1].strip()
                        if "/100" in score_text or "/ 100" in score_text:
                            score_value = int(score_text.split("/")[0].strip())
                            percentage = score_value

                            # Determine color based on score
                            if score_value >= 80:
                                color = "#28a745"  # Green for high scores
                            elif score_value >= 60:
                                color = "#ffc107"  # Yellow for medium scores
                            else:
                                color = "#dc3545"  # Red for low scores

                            section_content.append(f'''
                            <div class="overall-score">
                                <div class="score-label">{line.split(":")[0].strip()}:</div>
                                <div class="score-container">
                                    <div class="score-bar" style="width: {percentage}%; background-color: {color};"></div>
                                </div>
                                <div class="score-value">{score_value}/100</div>
                            </div>
                            ''')
                        else:
                            section_content.append(f'<p class="highlight-text">{line}</p>')
                    except (ValueError, IndexError):
                        section_content.append(f'<p class="highlight-text">{line}</p>')
                else:
                    section_content.append(f'<p>{line}</p>')

    # Close any open elements
    if in_list:
        formatted_html += '</ul>'

    if current_section:
        formatted_html += f'<div class="section-content">{"".join(section_content)}</div></div>'

    formatted_html += '</div>'  # Close analysis-results

    return formatted_html


def analyze_with_claude(text):
    """Send text to Claude API for analysis with improved error handling and retry logic"""
    prompt = """
    You are a mining industry expert with extensive experience in reviewing feasibility studies. Analyze this feasibility study for a mining project and provide a detailed evaluation with specific metrics, figures, and technical insights.

    Rate the following aspects on a scale of 1-10 and provide detailed explanations with industry-standard metrics:

    1. Resource Quality & Certainty: 
       - Assess the confidence level of resource estimation methods (Measured, Indicated, Inferred categories)
       - Evaluate resource grades (e.g., g/t for gold, % for base metals) compared to industry averages
       - Review drilling density, continuity of mineralization, and geostatistical methods
       - Analyze cut-off grades and their justification
       - Assess QA/QC protocols and resource categorization
       - Comment on resource tonnage and contained metal amounts

    2. Technical Viability: 
       - Evaluate proposed mining methods (open-pit vs underground, mining techniques)
       - Assess ore extraction rates (tonnes per day/year)
       - Review processing technology selection (recovery rates, process flow)
       - Analyze strip ratio (for open-pit) or dilution factors (for underground)
       - Evaluate mine design parameters, geotechnical considerations
       - Review metallurgical test work, crushing/grinding/processing parameters
       - Assess mine life estimates and production schedules
       - Comment on equipment selection and maintenance schedules

    3. Economic Analysis: 
       - Review capital expenditure (CAPEX) breakdown by major component
       - Analyze operating costs (OPEX) per tonne or unit of metal
       - Assess revenue projections based on production profile and price assumptions
       - Evaluate Net Present Value (NPV), Internal Rate of Return (IRR), and payback period
       - Review sensitivity analyses (to metal prices, CAPEX, OPEX, recovery, grade)
       - Analyze All-In Sustaining Costs (AISC) per unit of production
       - Comment on tax assumptions and royalty structures
       - Assess depreciation schedules and working capital requirements

    4. Financial Structure: 
       - Assess total funding requirements and proposed capital structure
       - Review debt-equity ratios and proposed financing mechanisms
       - Evaluate debt service coverage ratios and loan terms
       - Analyze return metrics against industry benchmarks
       - Assess proposed hedging strategies if any
       - Review cash flow projections and financial model assumptions
       - Comment on ownership structure and joint venture terms if applicable

    5. Market Analysis: 
       - Evaluate commodity price assumptions vs current prices and forward curves
       - Review market studies, supply-demand dynamics, and market share projections
       - Assess potential offtake agreements and sales strategies
       - Analyze product specifications and marketability
       - Review transportation and logistics costs to market
       - Comment on currency exchange assumptions and hedging strategies
       - Assess market concentration risks and customer diversity

    6. Environmental & Social Impact: 
       - Review environmental impact assessments and baseline studies
       - Evaluate permitting status, timeline, and remaining regulatory hurdles
       - Assess water management plans (sources, consumption, treatment, discharge)
       - Review waste management approaches (tailings, waste rock, hazardous materials)
       - Analyze community engagement plans and social license considerations
       - Evaluate overall ESG (Environmental, Social, Governance) considerations
       - Comment on closure plans and associated costs
       - Assess carbon footprint and climate change adaptation strategies

    7. Infrastructure & Logistics: 
       - Evaluate power requirements, sources, and reliability
       - Assess water supply adequacy and security
       - Review transportation infrastructure (roads, rail, ports)
       - Analyze site accessibility and seasonal constraints
       - Evaluate communications infrastructure
       - Assess workforce accommodation and facilities
       - Comment on proximity to suppliers and services
       - Review critical infrastructure development timelines

    8. Project Timeline: 
       - Evaluate the realism of the development schedule
       - Review critical path analysis and potential bottlenecks
       - Assess milestone planning and contingency allowances
       - Analyze construction timelines vs industry norms for similar projects
       - Review commissioning schedule and ramp-up assumptions
       - Comment on permitting timeline assumptions

    9. Risk Assessment: 
       - Analyze the comprehensiveness of risk identification
       - Evaluate proposed mitigation strategies for key risks
       - Assess contingency planning and budget allowances
       - Review sensitivity to critical variables (grade, recovery, prices)
       - Evaluate geopolitical risk considerations
       - Comment on insurance coverage and force majeure provisions
       - Assess technical risks specific to the proposed mining method

    Then provide:
    1. An overall financing potential score (1-100)
    2. A detailed summary of key strengths with specific figures and metrics
    3. A comprehensive list of key weaknesses and potential red flags
    4. Specific, actionable recommendations to improve the bankability of the project
    5. An assessment of the likelihood of securing financing based on historical patterns for similar projects

    Feasibility Study Text:
    {text}
    """

    # Check if extracted text is actually an error message
    if text.startswith("Error:"):
        return text

    # If text is too short, it's probably not a real feasibility study
    if len(text) < 1000:
        return "Error: The extracted text is too short to be a proper feasibility study. Please ensure you're uploading a complete document."

    # Check if API key is configured
    if not CLAUDE_API_KEY:
        return "Error: Claude API key is not configured. Please set the CLAUDE_API_KEY environment variable."

    # Define models to try in order of preference
    models_to_try = [
        "claude-3-7-sonnet-20250219",  # Try newest model first
        "claude-3-5-sonnet-20240620",  # Fall back to older model
        "claude-3-opus-20240229"  # Final fallback to opus
    ]

    # Maximum retries
    max_retries = 2
    retry_delay = 3  # seconds

    for model in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to use model: {model}, attempt {attempt + 1}")

                headers = {
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": CLAUDE_API_KEY
                }

                # Limit text to a reasonable size - max 40k chars for reliability
                trimmed_text = text[:40000]

                # Warn if text was truncated
                if len(text) > 40000:
                    logger.warning(f"Text truncated from {len(text)} to 40000 characters")

                payload = {
                    "model": model,
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt.format(text=trimmed_text)}]
                }

                # Make the API request with detailed error handling
                start_time = time.time()
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=180  # Increased timeout for larger documents
                )
                end_time = time.time()

                logger.info(f"API request took {end_time - start_time:.2f} seconds")
                logger.info(f"Response status code: {response.status_code}")

                # Handle different response status codes with specific error messages
                if response.status_code == 200:
                    # Success case
                    response_data = response.json()

                    if "content" in response_data and len(response_data["content"]) > 0:
                        logger.info(f"Successfully received response from {model}")
                        return response_data["content"][0]["text"]
                    else:
                        logger.error(f"Unexpected response structure: {json.dumps(response_data)[:500]}")
                        continue  # Try next attempt or model

                elif response.status_code == 401:
                    return "Error: API authentication failed. Please check your API key."
                elif response.status_code == 403:
                    return "Error: API access forbidden. Your API key may not have permission to use this model."
                elif response.status_code == 404:
                    logger.warning(f"Model {model} not found, will try next model")
                    break  # Skip to next model
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded, waiting before retry")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, waiting before retry")
                    time.sleep(retry_delay)
                    continue
                else:
                    # For other errors, log details and continue with retries
                    logger.error(f"Unexpected status code: {response.status_code}")
                    logger.error(f"Response body: {response.text[:1000]}")

                    # Only retry if it makes sense for this status code
                    if response.status_code >= 500:  # Server errors are retryable
                        time.sleep(retry_delay)
                        continue
                    else:
                        break  # Client errors are not retryable, move to next model

            except requests.exceptions.Timeout:
                logger.error("Request timeout")
                time.sleep(retry_delay)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                time.sleep(retry_delay)
                continue
            except json.JSONDecodeError:
                logger.error(f"JSON decode error. Response text: {response.text[:500]}")
                time.sleep(retry_delay)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.error(traceback.format_exc())
                return f"Error: An unexpected error occurred: {e}"

    # If we've exhausted all models and retries
    return "Error: Unable to get a successful response from Claude API after multiple attempts with different models. Please try again later."


@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Endpoint for asking follow-up questions about the feasibility study"""
    try:
        data = request.json
        question = data.get('question', '')
        original_analysis = data.get('original_analysis', '')

        if not question or not original_analysis:
            return jsonify({"error": "Missing question or original analysis"}), 400

        # Check if API key is configured
        if not CLAUDE_API_KEY:
            return jsonify({"error": "Claude API key is not configured"}), 500

        prompt = f"""
        You are a mining industry expert. You previously analyzed a mining feasibility study and provided the following analysis:

        {original_analysis}

        Now, the user is asking a follow-up question about this analysis:

        "{question}"

        Please answer their question specifically and concisely, using information from your original analysis. 
        If answering requires information not found in the analysis, please indicate this clearly and provide your best professional assessment based on standard industry practices.
        Include relevant technical details, figures, and metrics where appropriate.
        """

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": CLAUDE_API_KEY
        }

        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            response_data = response.json()
            answer = response_data["content"][0]["text"]
            return jsonify({"answer": answer})
        else:
            return jsonify({"error": f"Error from Claude API: {response.text[:500]}"}), 500

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Verify file is a PDF
    if file and file.filename.lower().endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Log file details before saving
            file_size = 0
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer

            logger.info(f"Processing file: {filename}, size: {file_size / 1024 / 1024:.2f} MB")

            # Check file size before processing
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({
                    "error": f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024} MB"}), 413

            file.save(filepath)
            logger.info(f"File saved: {filepath}")

            # Extract text from PDF
            logger.info("Extracting text from PDF")
            pdf_text = extract_text_from_pdf(filepath)

            # Handle text extraction errors
            if pdf_text.startswith("Error:"):
                return jsonify({"error": pdf_text}), 400

            # If text extraction resulted in very little text
            if len(pdf_text) < 100:
                logger.warning("Insufficient text extracted from PDF")
                return jsonify({
                    "error": "Could not extract sufficient text from the PDF. The document might be scanned or contain only images."}), 400

            # Analyze with Claude
            logger.info("Sending text to Claude for analysis")
            analysis = analyze_with_claude(pdf_text)

            # Handle analysis errors
            if analysis.startswith("Error:"):
                return jsonify({"error": analysis}), 400

            # Format the analysis results
            formatted_analysis = format_analysis_results(analysis)

            # Clean up the temp file
            try:
                os.remove(filepath)
                logger.info(f"Temporary file removed: {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {str(e)}")

            # Return both plain text and formatted HTML
            return jsonify({
                "analysis": analysis,
                "formatted_analysis": formatted_analysis
            })

        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Error processing upload: {str(e)}"}), 500

    logger.warning("File must be a PDF")
    return jsonify({"error": "File must be a PDF. Please upload a PDF document."}), 400


@app.route('/test-api', methods=['GET'])
def test_api():
    """Simple endpoint to test if the Claude API connection is working"""
    if not CLAUDE_API_KEY:
        return jsonify({"error": "API key is not configured"}), 500

    try:
        # Simple test message
        test_message = "Hello, Claude. Please respond with 'API connection successful.'"

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": CLAUDE_API_KEY
        }

        # Try with the newest model first
        model = "claude-3-7-sonnet-20250219"

        payload = {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": test_message}]
        }

        # Make request
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )

        # Return results
        return jsonify({
            "status": "success" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model_tested": model,
            "api_key_configured": bool(CLAUDE_API_KEY),
            "response_preview": response.text[:300] if response.status_code == 200 else response.text
        })

    except Exception as e:
        logger.error(f"API test error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })


@app.route('/api-status', methods=['GET'])
def api_status():
    """Simple HTML page to test and display API status"""
    return render_template('api_status.html')


if __name__ == '__main__':
    # Check if API key is set at startup
    if not CLAUDE_API_KEY:
        logger.critical("CLAUDE_API_KEY environment variable is not set. The application will not function correctly.")
        print("WARNING: CLAUDE_API_KEY environment variable is not set. Set it before running the application.")
    else:
        print(f"API key configured: {CLAUDE_API_KEY[:8]}...{CLAUDE_API_KEY[-4:]}")

    # Get port from environment variable (Render sets this) or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Make sure to listen on all interfaces (0.0.0.0) not just localhost
    app.run(host='0.0.0.0', port=port, debug=True)