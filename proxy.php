<?php
header("Access-Control-Allow-Origin: https://search.library.stonybrook.edu");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");

$query = $_REQUEST['query'] ?? '';

if (empty($query)) {
    http_response_code(400);
    echo 'Error: No query provided.';
    exit;
}

$escaped_query = escapeshellarg($query);
$command = "python ag.py url $escaped_query";  // use 'python3' if necessary
$output = shell_exec($command);  // capture error output as well
$url = trim($output);

// Validate and redirect
if (!$url || !preg_match('/^https?:\/\//', $url)) {
    http_response_code(500);
    echo 'Error: ag.py did not return a valid URL.';
    exit;
}

header("Content-Type: text/plain");
echo $url;
exit;
?>
