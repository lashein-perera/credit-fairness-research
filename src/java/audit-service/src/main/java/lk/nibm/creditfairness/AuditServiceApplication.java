package lk.nibm.creditfairness;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Fairness Audit Service.
 *
 * Thin Java layer over the Python research pipeline. Exposes credit scoring,
 * fairness/leakage auditing and per-applicant explanations over REST.
 */
@SpringBootApplication
public class AuditServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(AuditServiceApplication.class, args);
    }
}
