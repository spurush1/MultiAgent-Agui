#!/bin/bash
# Multi-Agent System Comprehensive Test Script
# Tests 30+ queries across BOM, Supplier, Materials, and Orchestrator agents

set -e

ORCHESTRATOR_URL="http://localhost:8013"
BOM_URL="http://localhost:8014"
SUPPLIER_URL="http://localhost:8011"
MATERIALS_URL="http://localhost:8012"

echo "======================================"
echo "Multi-Agent System Test Suite"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

test_count=0
success_count=0

# Function to test an endpoint
test_query() {
    local test_num=$1
    local description=$2
    local url=$3
    local data=$4
    
    ((test_count++))
    echo -e "${BLUE}Test $test_num: $description${NC}"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        echo "$body" | jq -C '.' 2>/dev/null || echo "$body"
        ((success_count++))
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        echo "$body"
    fi
    echo ""
}

echo "======================================"
echo "Category 1: BOM Agent Tests (10 tests)"
echo "======================================"
echo ""

test_query 1 "Get BOM for 1.5L Turbocharged 4-Cylinder engine" \
    "$BOM_URL/get-bom" \
    '{"part_name":"1.5L Turbocharged 4-Cylinder"}'

test_query 2 "Get BOM for 3.5L V6 Engine" \
    "$BOM_URL/get-bom" \
    '{"part_name":"3.5L V6 Engine"}'

test_query 3 "Get BOM for 5.3L V8 Engine" \
    "$BOM_URL/get-bom" \
    '{"part_name":"5.3L V8 Engine"}'

test_query 4 "Get BOM for 6-Speed Automatic Transmission" \
    "$BOM_URL/get-bom" \
    '{"part_name":"6-Speed Automatic Transmission"}'

test_query 5 "Get BOM for 8-Speed Automatic Transmission" \
    "$BOM_URL/get-bom" \
    '{"part_name":"8-Speed Automatic Transmission"}'

test_query 6 "Get BOM for CVT Transmission" \
    "$BOM_URL/get-bom" \
    '{"part_name":"CVT Transmission"}'

test_query 7 "Get BOM for Hybrid Powertrain" \
    "$BOM_URL/get-bom" \
    '{"part_name":"Hybrid Powertrain"}'

test_query 8 "Get BOM for 2.5L 4-Cylinder Engine" \
    "$BOM_URL/get-bom" \
    '{"part_name":"2.5L 4-Cylinder Engine"}'

test_query 9 "Get BOM for 10-Speed Automatic Transmission" \
    "$BOM_URL/get-bom" \
    '{"part_name":"10-Speed Automatic Transmission"}'

test_query 10 "Get BOM for 6.2L V8 Engine" \
    "$BOM_URL/get-bom" \
    '{"part_name":"6.2L V8 Engine"}'

echo "======================================"
echo "Category 2: Supplier Risk Analysis (10 tests)"
echo "======================================"
echo ""

test_query 11 "Analyze risk for Denso in Japan" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Denso","country":"Japan"}'

test_query 12 "Analyze risk for Bosch in Germany" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Bosch","country":"Germany"}'

test_query 13 "Analyze risk for BorgWarner in USA" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"BorgWarner","country":"USA"}'

test_query 14 "Analyze risk for Continental AG in Germany" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Continental AG","country":"Germany"}'

test_query 15 "Analyze risk for Magna International in Canada" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Magna International","country":"Canada"}'

test_query 16 "Analyze risk for Aisin in Japan" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Aisin","country":"Japan"}'

test_query 17 "Analyze risk for Valeo in France" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Valeo","country":"France"}'

test_query 18 "Analyze risk for Delphi Technologies in USA" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Delphi Technologies","country":"USA"}'

test_query 19 "Analyze risk for ZF Friedrichshafen in Germany" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"ZF Friedrichshafen","country":"Germany"}'

test_query 20 "Analyze risk for Tenneco in USA" \
    "$SUPPLIER_URL/analyze-risk" \
    '{"supplier_name":"Tenneco","country":"USA"}'

echo "======================================"
echo "Category 3: Materials Discovery (10 tests)"
echo "======================================"
echo ""

test_query 21 "Find details about brake pads" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"brake pads"}'

test_query 22 "Find details about spark plugs" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"spark plugs"}'

test_query 23 "Find details about turbochargers" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"turbocharger"}'

test_query 24 "Find details about fuel injectors" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"fuel injector"}'

test_query 25 "Find details about brake calipers" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"brake caliper"}'

test_query 26 "Find details about alternator" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"alternator"}'

test_query 27 "Find details about catalytic converter" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"catalytic converter"}'

test_query 28 "Find details about radiator" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"radiator"}'

test_query 29 "Find details about shock absorber" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"shock absorber"}'

test_query 30 "Find details about ABS module" \
    "$MATERIALS_URL/find-material" \
    '{"part_name":"ABS control module"}'

echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "Total Tests: $test_count"
echo -e "${GREEN}Passed: $success_count${NC}"
echo -e "${RED}Failed: $((test_count - success_count))${NC}"
echo -e "Success Rate: $(echo "scale=2; $success_count * 100 / $test_count" | bc)%"
echo ""

if [ $success_count -eq $test_count ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi
