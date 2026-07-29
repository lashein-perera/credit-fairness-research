package lk.nibm.creditfairness.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * Week 6 deliverable.
 *
 *   POST /api/score            applicant features -> decision + probability
 *   POST /api/audit            dataset -> fairness + leakage audit report
 *   GET  /api/explain/{id}     per-applicant SHAP explanation
 */
@RestController
@RequestMapping("/api")
public class ScoringController {

    @PostMapping("/score")
    public ResponseEntity<Map<String, Object>> score(@RequestBody Map<String, Object> applicant) {
        throw new UnsupportedOperationException("Week 6");
    }

    @PostMapping("/audit")
    public ResponseEntity<Map<String, Object>> audit(@RequestBody Map<String, Object> request) {
        throw new UnsupportedOperationException("Week 6");
    }

    @GetMapping("/explain/{id}")
    public ResponseEntity<Map<String, Object>> explain(@PathVariable String id) {
        throw new UnsupportedOperationException("Week 6");
    }
}
