package com.university.grades.service;

import com.university.grades.model.Student;
import com.university.grades.repository.StudentRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class StudentServiceTest {

    @Mock
    private StudentRepository studentRepository;

    @InjectMocks
    private StudentService studentService;

    private Student student1;
    private Student student2;

    @BeforeEach
    void setUp() {
        student1 = new Student("Alice", 8.5);
        student1.setId(1L);

        student2 = new Student("Bob", 6.0);
        student2.setId(2L);
    }

    @Test
    void getAllStudents_shouldReturnAllStudents() {
        when(studentRepository.findAll()).thenReturn(Arrays.asList(student1, student2));

        List<Student> result = studentService.getAllStudents();

        assertEquals(2, result.size());
        assertEquals("Alice", result.get(0).getName());
        assertEquals("Bob", result.get(1).getName());
        verify(studentRepository, times(1)).findAll();
    }

    @Test
    void createStudent_shouldSaveAndReturnStudent() {
        Student newStudent = new Student("Charlie", 9.0);
        when(studentRepository.save(newStudent)).thenReturn(newStudent);

        Student result = studentService.createStudent(newStudent);

        assertNotNull(result);
        assertEquals("Charlie", result.getName());
        assertEquals(9.0, result.getGrade());
        verify(studentRepository, times(1)).save(newStudent);
    }

    @Test
    void getGradeById_shouldReturnGradeWhenStudentExists() {
        when(studentRepository.findById(1L)).thenReturn(Optional.of(student1));

        Optional<Double> result = studentService.getGradeById(1L);

        assertTrue(result.isPresent());
        assertEquals(8.5, result.get());
    }

    @Test
    void getGradeById_shouldReturnEmptyWhenStudentNotFound() {
        when(studentRepository.findById(99L)).thenReturn(Optional.empty());

        Optional<Double> result = studentService.getGradeById(99L);

        assertFalse(result.isPresent());
    }

    @Test
    void getStudentById_shouldReturnStudentWhenExists() {
        when(studentRepository.findById(1L)).thenReturn(Optional.of(student1));

        Optional<Student> result = studentService.getStudentById(1L);

        assertTrue(result.isPresent());
        assertEquals("Alice", result.get().getName());
    }
}
